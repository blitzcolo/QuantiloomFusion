"""Torch-free support for the zero-shot public fusion models.

Where the third-party code clones and author weights live (``build/models/``,
gitignored, never committed; override with ``QUANTILOOM_ZEROSHOT_MODELS``), file
hashing, and the input preparation the wrappers in ``fusion.zeroshot`` share.
Nothing here imports torch, so it can be used and tested without it.

Pixel convention: package pixel-index coordinates (pixel centres on integers,
origin at the top-left pixel centre); the camera's continuous coordinates put
that centre at (0.5, 0.5), hence the +-0.5 in the pure-scale map.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..dataset.types import CameraGeometry
from ..geometry.camera import pixel_grid


# UMF-CMGR (Wang et al., IJCAI 2022): clone directory under models_dir() and the three author
# weight files the full test chain needs (CPSTN generator, MRRN registration, DIFN fusion).
UMF_CMGR_DIR = "UMF-CMGR"
UMF_CMGR_WEIGHTS = {
    "cpstn": "CPSTN/checkpoints/pretrained/latest_net_G.pth",
    "mrrn": "weights/reg_0280.pth",
    "difn": "weights/fus_0280.pth",
}



@dataclass(frozen=True)
class ZeroShotModel:
    """A registered zero-shot model's clone directory (under models_dir()), author weight files
    (relative to the clone) and a tag of our input preparation (bump it when that changes)."""

    code_dir: str
    weights: tuple[str, ...]
    preprocessing: str


_SUPERFUSION_PREP = ("superfusion-v1: VIS display sRGB round(x*255) as RGB; LWIR window round(x*255), "
                     "pure-scale bilinear onto the VIS frame, grey to 3 channels; output RGB, scored as BT.601 Y")
ZEROSHOT_MODELS: dict[str, ZeroShotModel] = {
    "superfusion_msrs": ZeroShotModel("SuperFusion", ("checkpoint/MSRS.pth",), _SUPERFUSION_PREP),
    "superfusion_roadscene": ZeroShotModel("SuperFusion", ("checkpoint/RoadScene.pth",), _SUPERFUSION_PREP),
    "umf_cmgr": ZeroShotModel(
        UMF_CMGR_DIR, tuple(UMF_CMGR_WEIGHTS.values()),
        "umf-cmgr-v1: native VIS frame; CPSTN on VIS display round(x*255) in [-1,1], pseudo-IR truncated to "
        "8 bit; MRRN(pseudo-IR, LWIR window round(x*255) pure-scale onto the VIS frame); DIFN(registered "
        "LWIR, BT.601 grey of the 8-bit VIS); output clipped to [0,1]"),
}


def models_dir() -> Path:
    """Root of the third-party model clones: ``QUANTILOOM_ZEROSHOT_MODELS`` or ``<repo>/build/models``."""
    env = os.environ.get("QUANTILOOM_ZEROSHOT_MODELS")
    return Path(env) if env else Path(__file__).resolve().parents[3] / "build" / "models"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quantise_8bit(a: np.ndarray) -> np.ndarray:
    """[0, 1] -> uint8 by round(x * 255) after clipping (an 8-bit image file, as the authors read)."""
    return np.rint(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)


def canvas_to_native(px: np.ndarray, py: np.ndarray, ref: CameraGeometry,
                     src: CameraGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates on the reference-size canvas -> native source pixel coordinates.

    Pure scale about the principal points (the ``ResizeRegistrar`` form):
    u_src = (u_ref + 0.5 - c_ref) * f_src / f_ref + c_src - 0.5.
    """
    nx = (px + 0.5 - ref.cx) * (src.fx / ref.fx) + src.cx - 0.5
    ny = (py + 0.5 - ref.cy) * (src.fy / ref.fy) + src.cy - 0.5
    return nx, ny


def pure_scale_map(ref: CameraGeometry, src: CameraGeometry) -> np.ndarray:
    """(H_ref, W_ref, 2) native source coordinates of every reference pixel under the pure-scale map."""
    grid = pixel_grid(ref)
    return np.stack(canvas_to_native(grid[..., 0], grid[..., 1], ref, src), axis=-1)


def lwir_canvas_8bit(lwir: np.ndarray, ref: CameraGeometry, src: CameraGeometry) -> np.ndarray:
    """The normalised LWIR image quantised to 8 bit and resampled onto the reference frame by the
    pure-scale map (bilinear, 0 outside the LWIR image): the model's own matcher does the rest."""
    lwir8 = quantise_8bit(np.asarray(lwir, np.float32))
    rmap = pure_scale_map(ref, src)
    return cv2.remap(lwir8, rmap[..., 0].astype(np.float32), rmap[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def umf_cmgr_inputs(vis_rgb: np.ndarray, lwir: np.ndarray, ref: CameraGeometry,
                    src: CameraGeometry) -> dict[str, np.ndarray]:
    """UMF-CMGR inputs on the native VIS frame, as the authors' test code reads its image files.

    Test/test_reg_fusion.py and dataloader/joint_data.py read every image with
    ``cv2.IMREAD_GRAYSCALE`` and divide by 255; CPSTN (test.py, single dataset,
    ``--preprocess none``) reads the visible image as RGB and normalises it to [-1, 1].
    Here the files are replaced by their 8-bit contents:

    * ``vis_8bit``: (H, W, 3) uint8, the display sRGB image quantised by round(x * 255);
    * ``cpstn_rgb``: (3, H, W) float32 in [-1, 1], CPSTN's input (x / 255 - 0.5) / 0.5;
    * ``vi``: (H, W) float32 in [0, 1], BT.601 grey of ``vis_8bit`` (OpenCV's conversion) / 255;
    * ``ir``: (H, W) float32 in [0, 1], the 8-bit LWIR on the VIS frame by the pure-scale map / 255.

    H and W must be multiples of 4: MRRN halves the frame twice and doubles the flow back, and
    CPSTN would otherwise resize the image (its make_power_2 rule), which we do not do.
    """
    vis = np.asarray(vis_rgb, np.float32)
    if vis.ndim != 3 or vis.shape[2] != 3:
        raise ValueError(f"expected a (H, W, 3) display image, got {vis.shape}")
    h, w = vis.shape[:2]
    if (w, h) != (ref.width, ref.height):
        raise ValueError(f"VIS image {w}x{h} does not match its geometry {ref.width}x{ref.height}")
    if h % 4 or w % 4:
        raise ValueError(f"UMF-CMGR needs a frame whose sides are multiples of 4, got {w}x{h}")
    vis8 = quantise_8bit(vis)
    grey8 = cv2.cvtColor(vis8, cv2.COLOR_RGB2GRAY)
    ir8 = lwir_canvas_8bit(lwir, ref, src)
    return {
        "vis_8bit": vis8,
        "cpstn_rgb": (vis8.transpose(2, 0, 1).astype(np.float32) / 255.0 - 0.5) / 0.5,
        "vi": grey8.astype(np.float32) / 255.0,
        "ir": ir8.astype(np.float32) / 255.0,
    }


def git_head(repo: Path) -> str:
    """Commit checked out in a clone, read from its .git directory (no git process): HEAD, the loose
    ref it names, or packed-refs; "unknown" when the clone has no readable .git."""
    git = Path(repo) / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="ascii").strip()
    except OSError:
        return "unknown"
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    loose = git / ref
    if loose.is_file():
        return loose.read_text(encoding="ascii").strip()
    packed = git / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="ascii").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    return "unknown"


def model_identity(name: str) -> dict[str, str]:
    """What identifies a zero-shot model's results: name, clone commit, SHA-256 of every author
    weight file (``relpath=sha256`` joined by ";") and the preprocessing tag.

    Raises KeyError for an unknown name and FileNotFoundError when the clone or a weight file is
    missing, so a run without the assets fails loudly instead of skipping the model.
    """
    try:
        model = ZEROSHOT_MODELS[name]
    except KeyError:
        raise KeyError(f"unknown zero-shot model {name!r}; known: {sorted(ZEROSHOT_MODELS)}") from None
    root = models_dir() / model.code_dir
    missing = [rel for rel in model.weights if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"{name}: author weights missing under {root}: {missing}")
    return {"model": name, "code_commit": git_head(root),
            "weights_sha256": ";".join(f"{rel}={file_sha256(root / rel)}" for rel in model.weights),
            "preprocessing": model.preprocessing}
