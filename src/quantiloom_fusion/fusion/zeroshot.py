"""Zero-shot public deep fusion models as FusionMethods.

* SuperFusion (Tang et al., JAS 2022): ``superfusion_msrs``, ``superfusion_roadscene``.
* UMF-CMGR (Wang et al., IJCAI 2022): ``umf_cmgr``.

The authors' code and released weights live under build/models/ (gitignored,
never committed); this module only drives them. Each model registers the
infrared image onto the visible one itself and then fuses, so the methods take
unregistered VIS + LWIR and ignore x.warps; each reports the warp it applied
(VIS grid -> native LWIR pixels) in ``extras["warps"]``. Import this module
explicitly: it needs torch and is not imported by quantiloom_fusion.fusion.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
import time
import types

import numpy as np
import torch

from ..core.bands import Band
from ..geometry.warp import Warp
from .base import Capabilities, FusionInput, FusionMethod, FusionOutput
from .registry import register
from .zeroshot_assets import (
    UMF_CMGR_DIR,
    UMF_CMGR_WEIGHTS,
    canvas_to_native,
    file_sha256,
    lwir_canvas_8bit,
    models_dir,
    quantise_8bit,
    umf_cmgr_inputs,
)

__optional_deps__ = ("torch",)

MODELS_DIR = models_dir()
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


def _to_ycrcb(rgb: torch.Tensor):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.clamp(0, 1), ((b - y) * 0.564 + 0.5).clamp(0, 1), ((r - y) * 0.713 + 0.5).clamp(0, 1)


def _from_ycbcr(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
    cr_c, cb_c = cr - 0.5, cb - 0.5
    return torch.cat([y + 1.403 * cr_c, y - 0.714 * cr_c - 0.344 * cb_c, y + 1.773 * cb_c], dim=1).clamp(0, 1)


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
        self.weights_sha256 = file_sha256(path)
        modules = _shim().import_modules(str(CODE_DIR))
        self._dm = modules.DenseMatcher()
        self._fn = modules.FusionNet()
        self._st = modules.SpatialTransformer(256, 256, True)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._dm.load_state_dict(state["DM"])
        self._fn.load_state_dict(state["FN"])
        self._dm.cuda().eval()
        self._fn.cuda().eval()

    def fuse(self, x: FusionInput) -> FusionOutput:
        self.check_input(x)
        if x.reference != Band.VIS or set(x.images) != {Band.VIS, Band.LWIR}:
            raise ValueError("SuperFusion fuses exactly VIS (reference) + LWIR")
        gv, gl = x.geometry[x.reference], x.geometry[Band.LWIR]
        vis8 = quantise_8bit(np.asarray(x.images[Band.VIS], np.float32))
        ir8 = lwir_canvas_8bit(x.images[Band.LWIR], gv, gl)
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
        nx, ny = canvas_to_native(px, py, gv, gl)
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


# ---------------------------------------------------------------------------------------------
# UMF-CMGR


def umf_cmgr_assets_available() -> bool:
    root = models_dir() / UMF_CMGR_DIR
    return all((root / rel).is_file() for rel in UMF_CMGR_WEIGHTS.values())


def _import_umf_cmgr(code_dir: str):
    """Import the inference classes from the untouched UMF-CMGR clone.

    Import-time stubs, only where the real package is not installed and only for names that no
    inference path uses (nothing numerical goes through them):

    * ``matplotlib.pyplot``: imported at the top of models/layers.py, never called there;
    * ``torchvision``: imported at the top of CPSTN/models/networks.py, used only by the VGG19
      perceptual loss of CPSTN training (``torchvision.models.vgg19``).

    models/transform.py (kornia) is never imported. models/layers.py builds its sampling grids on
    CUDA when its module flag ``gpu_use`` is set; it is switched off so the grids are CPU buffers
    that move with the module (the numbers are the same).
    """
    stubs: dict[str, types.ModuleType] = {}
    if importlib.util.find_spec("matplotlib") is None:
        mpl, plt = types.ModuleType("matplotlib"), types.ModuleType("matplotlib.pyplot")
        mpl.pyplot = plt
        stubs |= {"matplotlib": mpl, "matplotlib.pyplot": plt}
    if importlib.util.find_spec("torchvision") is None:
        stubs["torchvision"] = types.ModuleType("torchvision")
    clashing = {k: v for k, v in sys.modules.items() if k == "models" or k.startswith("models.")}
    for name in clashing:
        del sys.modules[name]
    sys.path.insert(0, code_dir)
    sys.modules.update(stubs)
    try:
        layers = importlib.import_module("models.layers")
        layers.gpu_use = False
        deformable = importlib.import_module("models.deformable_net")
        fusion = importlib.import_module("models.fusion_net")
        spec = importlib.util.spec_from_file_location(
            "umf_cmgr_cpstn_networks", f"{code_dir}/CPSTN/models/networks.py")
        networks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(networks)
    finally:
        sys.path.remove(code_dir)
        for name in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
            del sys.modules[name]
        sys.modules.update(clashing)
        for name in stubs:
            sys.modules.pop(name, None)
    return types.SimpleNamespace(SpatialTransformer=layers.SpatialTransformer,
                                 DeformableNet=deformable.DeformableNet, FusionNet=fusion.FusionNet,
                                 networks=networks)


def _load_checked(net: torch.nn.Module, state: dict, what: str) -> None:
    """Load with the base nn.Module implementation and fail on any mismatch.

    DeformableNet overrides load_state_dict with strict=False (and pops its grids), which would hide
    a mismatch; only the non-learned sampling-grid buffers (``*.grid``) may be missing.
    """
    res = torch.nn.Module.load_state_dict(net, state, strict=False)
    missing = [k for k in res.missing_keys if not k.endswith(".grid")]
    unexpected = [k for k in res.unexpected_keys if not k.endswith(".grid")]
    if missing or unexpected:
        raise RuntimeError(f"{what}: weights do not match the model (missing {missing}, unexpected {unexpected})")


@contextlib.contextmanager
def _determinism(strict: bool):
    """cudnn.benchmark off, cudnn.deterministic on, torch.use_deterministic_algorithms (strict or
    warn-only); the previous global settings are restored afterwards."""
    cudnn = torch.backends.cudnn
    prev = (cudnn.benchmark, cudnn.deterministic, torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled())
    cudnn.benchmark, cudnn.deterministic = False, True
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    try:
        yield
    finally:
        cudnn.benchmark, cudnn.deterministic = prev[0], prev[1]
        torch.use_deterministic_algorithms(prev[2], warn_only=prev[3])


@register("umf_cmgr")
class UmfCmgr(FusionMethod):
    """UMF-CMGR's full test chain on an unregistered VIS + LWIR pair (Test/test_reg_fusion.py).

    The authors' convention: CPSTN translates the visible RGB image into a pseudo-infrared image
    (the ``it`` input), the registration network MRRN (``DeformableNet``) registers the infrared
    image to that pseudo-infrared image -- a mono-modal problem -- and the fusion network DIFN
    (``FusionNet``, 64 features) fuses the registered infrared with the visible grey image. The
    pseudo-infrared image passes through an 8-bit file in the authors' pipeline (CPSTN's
    tensor2im truncates (x + 1) / 2 * 255 to uint8), reproduced here. Inputs are prepared by
    ``zeroshot_assets.umf_cmgr_inputs`` on the native VIS frame (MRRN's ``shape`` option: its
    sampling grids are rebuilt for the frame); LWIR goes onto that frame by the pure-scale map.

    Output: the fused luminance (H, W) in [0, 1] (DIFN's tanh output clipped to [0, 1], as the
    authors' 8-bit image writer saturates, but not quantised), on the VIS grid. Inference is
    deterministic: see ``_determinism``; ``extras["determinism"]`` says whether the strict mode ran.
    """

    capabilities = Capabilities(
        bands=frozenset({Band.VIS, Band.LWIR}),
        needs_registration=False,
        heterogeneous_resolution=True,
        color_output=False,
        trainable=False,
    )

    def __init__(self, device: str | None = None) -> None:
        root = models_dir() / UMF_CMGR_DIR
        paths = {k: root / rel for k, rel in UMF_CMGR_WEIGHTS.items()}
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"UMF-CMGR weights missing: {missing}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights_sha256 = {UMF_CMGR_WEIGHTS[k]: file_sha256(p) for k, p in paths.items()}
        mods = _import_umf_cmgr(str(root))
        self._mods = mods
        net = mods.networks
        # CPSTN test options: input_nc 3, output_nc 1, ngf 64, resnet_9blocks, instance norm, no dropout.
        self._cpstn = net.ResnetGenerator(3, 1, 64, norm_layer=net.get_norm_layer("instance"),
                                          use_dropout=False, n_blocks=9)
        self._mrrn = mods.DeformableNet()
        self._difn = mods.FusionNet(nfeats=64)
        _load_checked(self._cpstn, torch.load(paths["cpstn"], map_location="cpu", weights_only=True), "CPSTN")
        _load_checked(self._mrrn, torch.load(paths["mrrn"], map_location="cpu", weights_only=True)["net"], "MRRN")
        _load_checked(self._difn, torch.load(paths["difn"], map_location="cpu", weights_only=True)["net"], "DIFN")
        for m in (self._cpstn, self._mrrn, self._difn):
            m.to(self.device).eval()
        self._frame: tuple[int, int] | None = None

    def _set_frame(self, h: int, w: int) -> None:
        """What DeformableNet.forward(..., shape=(h, w)) does: sampling grids for the frame and its half."""
        if self._frame != (h, w):
            st = self._mods.SpatialTransformer
            self._mrrn.spatial_transform = st([h, w]).to(self.device)
            self._mrrn.spatial_transform_f = st([int(h / 2), int(w / 2)]).to(self.device)
            self._frame = (h, w)

    def _forward(self, cpstn_rgb: torch.Tensor, vi: torch.Tensor, ir: torch.Tensor):
        with torch.no_grad():
            it = self._cpstn(cpstn_rgb)
            it = torch.floor(((it + 1.0) / 2.0 * 255.0).clamp(0.0, 255.0)) / 255.0
            ir_pred, _f_warp, flow, _f1, _f2, _disp = self._mrrn(it, ir)
            fused = self._difn(ir_pred, vi)
        return it, ir_pred, flow, fused

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def fuse(self, x: FusionInput) -> FusionOutput:
        self.check_input(x)
        if x.reference != Band.VIS or set(x.images) != {Band.VIS, Band.LWIR}:
            raise ValueError("UMF-CMGR fuses exactly VIS (reference) + LWIR")
        gv, gl = x.geometry[Band.VIS], x.geometry[Band.LWIR]
        prep = umf_cmgr_inputs(x.images[Band.VIS], x.images[Band.LWIR], gv, gl)
        h, w = prep["vi"].shape
        self._set_frame(h, w)
        cp = torch.from_numpy(prep["cpstn_rgb"])[None].to(self.device)
        vi = torch.from_numpy(prep["vi"])[None, None].to(self.device)
        ir = torch.from_numpy(prep["ir"])[None, None].to(self.device)
        mode = "strict"
        self._sync()
        t0 = time.perf_counter()
        try:
            with _determinism(strict=True):
                it, ir_pred, flow, fused = self._forward(cp, vi, ir)
        except RuntimeError as e:
            if "deterministic" not in str(e):
                raise
            mode = "warn_only"
            with _determinism(strict=False):
                it, ir_pred, flow, fused = self._forward(cp, vi, ir)
        self._sync()
        runtime = time.perf_counter() - t0
        # SpatialTransformer: sample location = grid + flow, channel 0 = row (y), 1 = column (x), in
        # pixel-index units of the VIS-size canvas (align_corners=True normalisation).
        f = flow[0].double().cpu().numpy()
        py = np.arange(h, dtype=np.float64)[:, None] + f[0]
        px = np.arange(w, dtype=np.float64)[None, :] + f[1]
        nx, ny = canvas_to_native(px, py, gv, gl)
        valid = (nx >= 0) & (nx <= gl.width - 1) & (ny >= 0) & (ny <= gl.height - 1)
        warp = Warp(Band.VIS, Band.LWIR, np.stack([nx, ny], axis=-1), valid,
                    {"method": "umf_cmgr_mrrn", "weights_sha256": dict(self.weights_sha256)})
        return FusionOutput(
            fused[0, 0].clamp(0.0, 1.0).cpu().numpy().astype(np.float32),
            {
                "warps": {Band.LWIR: warp},
                "runtime_s": runtime,
                "determinism": {"cudnn_benchmark": False, "cudnn_deterministic": True,
                                "use_deterministic_algorithms": mode},
                "device": str(self.device),
                "pseudo_ir": it[0, 0].cpu().numpy(),
                "resized_lwir": prep["ir"],
                "registered_lwir": ir_pred[0, 0].cpu().numpy(),
                "vis_8bit": prep["vis_8bit"],
            },
        )
