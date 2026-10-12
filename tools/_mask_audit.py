# Masked tapp-vs-truth audit for the smoke dataset (peer review v4, issue 1).
# Usage: python _mask_audit.py <dataset_root_winpath>
import json, sys, glob, os
import numpy as np
import OpenEXR

root = sys.argv[1]

def read(path):
    f = OpenEXR.File(path)
    ch = f.channels()
    if "RGBA" in ch:
        return ch["RGBA"].pixels[:, :, 0].astype(np.float64)
    if "Channel_0" in ch:
        return ch["Channel_0"].pixels.astype(np.float64)
    key = list(ch.keys())[0]
    px = ch[key].pixels
    return px.astype(np.float64) if px.ndim == 2 else px[:, :, 0].astype(np.float64)

for scene in ["concrete_a000_e00_r00_n00", "concrete_night_a000_e00_r00_n00",
              "concrete_moonlight_a000_e00_r00_n00"]:
    for band in ["mwir", "lwir"]:
        base = glob.glob(os.path.join(root, "samples", scene, band, "v*", ""))
        if not base:
            continue
        base = sorted(base)[-1]
        tapp = read(base + "frame_tapp.exr")
        tsurf = read(base + "frame_truth_surface_temperature.exr")
        valid = read(base + "frame_truth_validity.exr")
        matid = read(base + "frame_truth_material_id.exr")
        mj = os.path.join(base, "frame_truth_materials.json")
        mats = json.load(open(mj)) if os.path.exists(mj) else []
        names = {m.get("material_id", i + 1): m.get("name", "?") for i, m in enumerate(mats)} \
            if mats and isinstance(mats, list) else {}

        print(f"\n=== {scene} / {band} === shape={tapp.shape}")
        for label, mask in [("ALL      ", np.ones(tapp.shape, bool)),
                            ("miss/sky ", valid == 0),
                            ("opaque   ", valid == 1),
                            ("partial  ", valid == 3),
                            ("transmit ", valid == 4)]:
            n = int(mask.sum())
            if n == 0:
                print(f"  {label} n=0")
                continue
            t, s = tapp[mask], tsurf[mask]
            print(f"  {label} n={n:5d}  tapp mean={t.mean():7.2f} min={t.min():7.2f} max={t.max():7.2f} | "
                  f"truth mean={s.mean():7.2f} | tapp-truth={t.mean()-s.mean():7.2f}")
        # per-material on opaque pixels
        op = valid == 1
        for mid in np.unique(matid[op]):
            m = op & (matid == mid)
            t, s = tapp[m], tsurf[m]
            nm = names.get(int(mid), f"id{int(mid)}")
            print(f"    mat {int(mid):3d} {nm[:44]:46s} n={int(m.sum()):4d} tapp={t.mean():7.2f} truth={s.mean():7.2f} diff={t.mean()-s.mean():7.2f}")
