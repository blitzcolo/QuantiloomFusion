"""Zero-shot SuperFusion (Tang et al., JAS 2022) as a FusionMethod.

The authors' code and released weights live under build/models/ (gitignored,
never committed); this module only drives them. SuperFusion registers the
infrared image onto the visible one itself (dense matcher) and then fuses, so
the method takes unregistered VIS + LWIR and ignores x.warps. Import this
module explicitly: it needs torch and a CUDA device and is not imported by
quantiloom_fusion.fusion.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ..core.bands import Band
from ..geometry.camera import pixel_grid
from ..geometry.warp import Warp
from .base import Capabilities, FusionInput, FusionMethod, FusionOutput
from .registry import register

__optional_deps__ = ("torch",)

MODELS_DIR = Path(os.environ.get("QUANTILOOM_ZEROSHOT_MODELS", Path(__file__).resolve().parents[3] / "build" / "models"))
CODE_DIR = MODELS_DIR / "SuperFusion"
PREPROCESSING = {
    "vis": "display sRGB, quantised to 8 bit (round(x*255)), fed as RGB in [0,1]",
    "lwir": "8-bit grey replicated to 3 channels, bilinear pure-scale resize onto the VIS frame",
    "resize": "u_src = (u_ref + 0.5 - c_ref) * f_src / f_ref + c_src - 0.5 (ResizeRegistrar form)",
}


def assets_available(checkpoint: str = "MSRS") -> bool:
    return (CODE_DIR / "checkpoint" / f"{checkpoint}.pth").is_file() and torch.cuda.is_available()


def _shim():
    spec = importlib.util.spec_from_file_location("superfusion_shim", MODELS_DIR / "superfusion_shim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_ycrcb(rgb: torch.Tensor):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.clamp(0, 1), ((b - y) * 0.564 + 0.5).clamp(0, 1), ((r - y) * 0.713 + 0.5).clamp(0, 1)


def _from_ycbcr(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
    cr_c, cb_c = cr - 0.5, cb - 0.5
    return torch.cat([y + 1.403 * cr_c, y - 0.714 * cr_c - 0.344 * cb_c, y + 1.773 * cb_c], dim=1).clamp(0, 1)


def _quantise(a: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)


class _SuperFusion(FusionMethod):
    capabilities = Capabilities(
        bands=frozenset({Band.VIS, Band.LWIR}),
        needs_registration=False,
        heterogeneous_resolution=True,
        color_output=True,
        trainable=False,
    )
    checkpoint: str

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SuperFusion's dense matcher builds CUDA tensors at construction; no CUDA device")
        path = CODE_DIR / "checkpoint" / f"{self.checkpoint}.pth"
        self.weights_sha256 = _sha256(path)
        modules = _shim().import_modules(str(CODE_DIR))
        self._dm = modules.DenseMatcher()
        self._fn = modules.FusionNet()
        self._st = modules.SpatialTransformer(256, 256, True)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._dm.load_state_dict(state["DM"])
        self._fn.load_state_dict(state["FN"])
        self._dm.cuda().eval()
        self._fn.cuda().eval()

    def _resize_map(self, x: FusionInput) -> np.ndarray:
        gv, gl = x.geometry[x.reference], x.geometry[Band.LWIR]
        grid = pixel_grid(gv)
        return np.stack(
            [
                (grid[..., 0] + 0.5 - gv.cx) * (gl.fx / gv.fx) + gl.cx - 0.5,
                (grid[..., 1] + 0.5 - gv.cy) * (gl.fy / gv.fy) + gl.cy - 0.5,
            ],
            axis=-1,
        )

    def fuse(self, x: FusionInput) -> FusionOutput:
        self.check_input(x)
        if x.reference != Band.VIS or set(x.images) != {Band.VIS, Band.LWIR}:
            raise ValueError("SuperFusion fuses exactly VIS (reference) + LWIR")
        vis8 = _quantise(np.asarray(x.images[Band.VIS], np.float32))
        lwir8 = _quantise(np.asarray(x.images[Band.LWIR], np.float32))
        rmap = self._resize_map(x)
        ir8 = cv2.remap(lwir8, rmap[..., 0].astype(np.float32), rmap[..., 1].astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        h, w = vis8.shape[:2]
        vi = torch.from_numpy(vis8).permute(2, 0, 1)[None].float().div(255).cuda()
        ir = torch.from_numpy(ir8)[None, None].float().div(255).repeat(1, 3, 1, 1).cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            disp = self._dm(ir, vi)["ir2vis"]
            ir_reg = self._st(ir, disp)
            vy, vcb, vcr = _to_ycrcb(vi)
            fused = _from_ycbcr(self._fn(ir_reg[:, 0:1], vy), vcb, vcr)
        torch.cuda.synchronize()
        runtime = time.perf_counter() - t0
        # grid_sample (align_corners=False) of u = linspace(-1,1) grid + disp: pixel = ((u + 1) * size - 1) / 2
        d = disp[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
        ux = np.linspace(-1.0, 1.0, w)[None, :] + d[..., 0]
        uy = np.linspace(-1.0, 1.0, h)[:, None] + d[..., 1]
        px, py = ((ux + 1) * w - 1) / 2, ((uy + 1) * h - 1) / 2
        gv, gl = x.geometry[x.reference], x.geometry[Band.LWIR]
        nx = (px + 0.5 - gv.cx) * (gl.fx / gv.fx) + gl.cx - 0.5
        ny = (py + 0.5 - gv.cy) * (gl.fy / gv.fy) + gl.cy - 0.5
        valid = (nx >= 0) & (nx <= gl.width - 1) & (ny >= 0) & (ny <= gl.height - 1)
        warp = Warp(Band.VIS, Band.LWIR, np.stack([nx, ny], axis=-1), valid,
                    {"method": "superfusion_dense_matcher", "checkpoint": self.checkpoint,
                     "weights_sha256": self.weights_sha256})
        return FusionOutput(
            fused[0].permute(1, 2, 0).cpu().numpy().astype(np.float32),
            {
                "warps": {Band.LWIR: warp},
                "runtime_s": runtime,
                "resized_lwir": ir8,
                "registered_lwir": ir_reg[0, 0].cpu().numpy(),
                "vis_8bit": vis8,
            },
        )


@register("superfusion_msrs")
class SuperFusionMSRS(_SuperFusion):
    checkpoint = "MSRS"


@register("superfusion_roadscene")
class SuperFusionRoadScene(_SuperFusion):
    checkpoint = "RoadScene"
