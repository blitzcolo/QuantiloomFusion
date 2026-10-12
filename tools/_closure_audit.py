"""Offline radiometric closure audit for a Quantiloom camera capture.

Re-runs the NN atmosphere inference in Python (stdlib + safetensors + numpy)
and compares, per opaque pixel,

    L_pred(lambda) = tau(λ,r)·[ eps(λ)·B(λ,T_s) + (1-eps(λ))·L_down(λ) ] + L_path(λ,r)

against the corrected detector signal converted to an apparent temperature.
The point: tau / L_path / L_down come from the same neural nets the renderer
baked, so a residual beyond quadrature tolerance means the sensor chain saw
a different atmosphere than the one it asked for -- exactly the class of bug
the 8-12 um endpoint collapse was.

Usage:
    python _closure_audit.py <attempt_dir>

The atmosphere model pack directory comes from the capture itself
(capture.toml's atmosphere.model_pack), so this runs on Windows and WSL.

attempt_dir contains frame_corrected.exr, frame_tapp.exr,
frame_truth_surface_temperature.exr, frame_truth_validity.exr,
frame_truth_material_id.exr, frame_truth_materials.json,
frame_truth_ray_distance.exr, and capture.toml.
"""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import OpenEXR
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantiloom_fusion.core.paths import native_path  # noqa: E402

# --- Planck, W m^-2 sr^-1 nm^-1 ---------------------------------------------
C1 = 1.191042972e-16  # 2 h c^2, W m^2 sr^-1
C2 = 1.438776877e-2   # h c / k, m K


def planck_nm(lam_nm: np.ndarray, t_k: np.ndarray) -> np.ndarray:
    lam_m = lam_nm * 1e-9
    x = C2 / (lam_m * t_k)
    return C1 / lam_m**5 / np.expm1(x) * 1e-9


# --- safetensors -------------------------------------------------------------
def load_safetensors(path):
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + n])
    meta = header.pop("__metadata__", {})
    tensors = {}
    for name, info in header.items():
        dt = {"F32": np.float32, "F64": np.float64, "U8": np.uint8,
              "I32": np.int32, "I64": np.int64}[info["dtype"]]
        b0, b1 = info["data_offsets"]
        arr = np.frombuffer(raw[8 + n + b0 : 8 + n + b1], dtype=dt)
        tensors[name] = arr.reshape(info["shape"])
    return meta, tensors


class Net:
    """AtmosNet + ResMLP replica (see src/libQuantiloom/atmos)."""

    def __init__(self, path):
        meta, self.t = load_safetensors(path)
        self.band = json.loads(meta["band_json"])
        self.model = json.loads(meta["model_json"])
        self.ispec = json.loads(meta["input_spec_json"])["entries"]
        self.targets = json.loads(meta["targets_json"])["rows"]
        self.feature_names = json.loads(meta["feature_names_json"])
        self.pca_mode = self.model["pca_mode"]
        self.opaque_delta = float(meta["opaque_delta"])
        self.delta_clamp = float(meta["delta_clamp"])
        self.K = int(self.band["K"])
        self.norms = []
        for i in range(len(self.targets)):
            p = f"norm.{i}."
            self.norms.append(
                dict(mean=self.t[p + "mean"].astype(np.float64),
                     std=self.t[p + "std"].astype(np.float64),
                     log_eps=self.t[p + "log_eps"].astype(np.float64),
                     log_mask=self.t[p + "log_mask"].astype(bool)))

    def _linear(self, x, prefix):
        w = self.t[prefix + ".weight"].astype(np.float64)
        b = self.t[prefix + ".bias"].astype(np.float64)
        return x @ w.T + b

    def forward(self, rows):
        """rows: [n, P] raw feature values -> [n, T*K] physical outputs."""
        # AssembleFeatures: encode each entry from its source column.
        feats = []
        for e in self.ispec:
            v = rows[:, e["col"]].astype(np.float64)
            if e["kind"] == "onehot":
                vals = np.asarray(e["values"])
                best = vals[np.argmin(np.abs(vals[None, :] - v[:, None]), axis=1)]
                feats.append((vals[None, :] == best[:, None]).astype(np.float64))
            else:
                if e["kind"] == "log":
                    v = np.log(np.maximum(v, 1e-300))
                elif e["kind"] == "cos_deg":
                    v = np.cos(np.deg2rad(v))
                u = 2.0 * (v - e["lo"]) / (e["hi"] - e["lo"]) - 1.0
                feats.append(np.clip(u, -1.0, 1.0)[:, None])
        x = np.concatenate(feats, axis=1).astype(np.float64)

        h = self._linear(x, "stem")
        for i in range(self.model["blocks"]):
            p = f"blocks.{i}."
            a = self._linear(h, p + "fc1")
            mean = a.mean(axis=1, keepdims=True)
            var = np.maximum((a**2).mean(axis=1, keepdims=True) - mean**2, 0.0)
            inv = 1.0 / np.sqrt(var + 1e-5)
            v = (a - mean) * inv
            v = v * self.t[p + "norm.weight"] + self.t[p + "norm.bias"]
            v = v / (1.0 + np.exp(-v))
            h = h + self._linear(v, p + "fc2")

        if self.pca_mode == "none":
            z = self._linear(h, "head")
        else:
            c = self._linear(h, "proj")
            if self.pca_mode == "head":
                z = self._linear(c, "head")
            else:
                z = c @ self.t["pca.basis"].T + self.t["pca.mean"]
        z = z.astype(np.float64)

        # Inverse transforms (AtmosNet::Infer).
        out = np.empty_like(z)
        for ti, (tgt, nr) in enumerate(zip(self.targets, self.norms)):
            v = z[:, ti * self.K : (ti + 1) * self.K] * nr["std"] + nr["mean"]
            if nr["log_mask"].any():
                m = nr["log_mask"]
                v[:, m] = np.exp(v[:, m]) - nr["log_eps"][m]
            if tgt["kind"] == "delta":
                v = np.clip(v, 0.0, self.delta_clamp)
            else:
                v = np.maximum(v, 0.0)
                if nr["log_mask"].any():
                    m = nr["log_mask"]
                    v[:, m] = np.minimum(v[:, m], nr["log_eps"][m] * 1e4 * 2.0)
            out[:, ti * self.K : (ti + 1) * self.K] = v
        return out

    def wavenumbers(self):
        b = self.band
        return b["v1_cm"] + b["dv_cm"] * np.arange(self.K)

    def channel_at(self, lam_nm):
        """Nearest wavenumber channel, matching the baker's single-lambda path."""
        nu = 1e7 / lam_nm
        b = self.band
        k = int(round((nu - b["v1_cm"]) / b["dv_cm"]))
        return int(np.clip(k, 0, self.K - 1))


def read_exr(path):
    f = OpenEXR.File(str(path))
    ch = f.channels()
    key = sorted(ch.keys())[0]
    px = ch[key].pixels
    a = np.asarray(px, dtype=np.float64)
    return a if a.ndim == 2 else a[:, :, 0]


def read_exr3(path, keys):
    f = OpenEXR.File(str(path))
    ch = f.channels()
    cols = [np.asarray(ch[k].pixels, dtype=np.float64) for k in keys]
    return np.stack([c[:, :, 0] if c.ndim == 3 else c for c in cols], axis=-1)


def fill_row(net, weather, sun_zen_deg, rel_az_deg, h1_km, view_zen_deg, range_km):
    row = np.zeros(len(net.feature_names))
    for i, n in enumerate(net.feature_names):
        row[i] = dict(
            atmos_model=weather["atmos_model"], ihaze=weather["ihaze"],
            icld=weather["icld"], vis_km=weather["vis_km"],
            rainrt_mm_h=weather["rainrt_mm_h"], t_ground_K=weather["t_ground_K"],
            rh=weather["rh"], p_hPa=weather["p_hPa"],
            h2o_scale=weather["h2o_scale"], sun_zenith_deg=sun_zen_deg,
            sun_rel_azimuth_deg=rel_az_deg, h1_km=h1_km, h2_km=h1_km,
            cos_view_zenith=math.cos(math.radians(view_zen_deg)),
            range_km=range_km)[n]
    return row


def main():
    attempt = Path(sys.argv[1])
    cfg = tomllib.loads((attempt / "capture.toml").read_text())
    if "model_pack" not in cfg["atmosphere"]:
        sys.exit("capture.toml has no atmosphere.model_pack")
    models = native_path(cfg["atmosphere"]["model_pack"])
    atmos = cfg["atmosphere"]
    weather = dict(
        atmos_model=float(atmos.get("atmos_model", 2.0)),
        ihaze=float(atmos.get("ihaze", 1.0)), icld=float(atmos.get("icld", 0.0)),
        vis_km=float(atmos.get("vis_km", 23.0)),
        rainrt_mm_h=float(atmos.get("rainrt_mm_h", 0.0)),
        t_ground_K=float(atmos.get("t_ground_K", 293.15)),
        rh=float(atmos.get("rh", 0.5)), p_hPa=float(atmos.get("p_hPa", 1013.25)),
        h2o_scale=float(atmos.get("h2o_scale", 1.0)))
    sun_dir = np.asarray(cfg["lighting"]["sun_direction"], dtype=float)
    sun_zen = math.degrees(math.acos(np.clip(sun_dir[1], -1.0, 1.0)))
    h1_km = float(atmos.get("h1_km", 0.1))
    w2m = float(cfg["scene"].get("world_units_to_meters", 1.0))
    sun_zen_bake = min(sun_zen, 85.0)
    night = sun_zen > 85.0
    scatter_scale = float(atmos.get("solar_scatter_scale", 1.0))

    tapp = read_exr(attempt / "frame_tapp.exr")
    truth = read_exr(attempt / "frame_truth_surface_temperature.exr")
    valid = read_exr(attempt / "frame_truth_validity.exr")
    mat = read_exr(attempt / "frame_truth_material_id.exr")
    dist = read_exr(attempt / "frame_truth_ray_distance.exr")
    mats = {m["material_id"]: m for m in
            json.loads((attempt / "frame_truth_materials.json").read_text())}

    lam_grid = np.array([8000.0, 9000.0, 10000.0, 11000.0, 12000.0])
    lam_w = np.array([0.5, 1.0, 1.0, 1.0, 0.5]) * 1000.0 / 4000.0  # trapezoid

    nets = {c: Net(models / f"lwir_ground_{c}.safetensors")
            for c in ("tau", "lpath", "ldown")}
    thermal_band = True  # lwir

    # Evaluate each net once per pixel at its own slant range (the baker's
    # ground path is horizontal: view zenith 90 deg, path range feature).
    ranges_km = np.clip(dist.ravel() * w2m / 1000.0, 0.05, 20.0)
    n_px = ranges_km.size

    def infer_at_channels(net, rows):
        out = net.forward(rows)
        return out[:, [net.channel_at(l) for l in lam_grid]]

    tau_rows = np.stack([
        fill_row(nets["tau"], weather, sun_zen_bake, 0.0, h1_km, 90.0, r)
        for r in ranges_km])
    lpath_rows = np.stack([
        fill_row(nets["lpath"], weather, sun_zen_bake, 0.0, h1_km, 90.0, r)
        for r in ranges_km])
    ldown_row = fill_row(nets["ldown"], weather, sun_zen_bake, 0.0, h1_km,
                         90.0, 1.0)[None, :]

    delta = infer_at_channels(nets["tau"], tau_rows)
    opaque = (delta.min(axis=1) > nets["tau"].opaque_delta)
    tau = np.exp(-delta)

    def nm_scale(net):
        lam = 1e7 / net.wavenumbers()
        ch = [net.channel_at(l) for l in lam_grid]
        return 1e11 / (lam[ch] ** 2)

    lp = infer_at_channels(nets["lpath"], lpath_rows) * nm_scale(nets["lpath"])
    # Single-product lpath in a non-thermal band would be scaled/zeroed for
    # night or scaled sun; LWIR lpath is pure thermal emission: untouched.
    _ = night, scatter_scale, thermal_band
    ld = infer_at_channels(nets["ldown"], ldown_row) * nm_scale(nets["ldown"])
    ld = np.repeat(ld, n_px, axis=0)

    eps = np.vectorize(
        lambda m: mats.get(int(m), {}).get("band_emissivity", 0.9))(mat.ravel())
    eps = np.clip(eps[:, None], 0.0, 1.0)

    ts = np.clip(truth.ravel(), 1.0, 5000.0)[:, None]
    bb = planck_nm(lam_grid[None, :], ts)

    l_pred = tau * (eps * bb + (1.0 - eps) * ld) + lp
    l_pred_bar = (l_pred * lam_w[None, :]).sum(axis=1)
    # Invert band-average radiance to apparent temperature (bisection).
    def invert(lb):
        lo, hi = 10.0, 5000.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            v = (planck_nm(lam_grid, mid) * lam_w).sum()
            if v < lb:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    t_pred = np.array([invert(v) for v in l_pred_bar])

    # Specular variant: for each pixel, replace the isotropic L_down by the
    # renderer's flat-slab sky model evaluated along the mirror-reflection
    # direction of the primary ray. The miss shader uses
    #   L_sky(theta) = B_air * [1 - (1 - eps0)^min(sec theta, 5)],
    #   eps0 = L_down_zenith / B_air,   B_air = Planck(T_air),
    # with T_air = lighting.atmosphere_temperature_k and the baked ldown as
    # the zenith term. Only pixels whose reflection leaves the ground plane
    # (cos zenith > 0) reflect sky; the rest hit scene geometry and are kept
    # under the isotropic model.
    t_air = float(cfg["lighting"].get("atmosphere_temperature_k", 284.68))
    b_air = planck_nm(lam_grid, t_air)
    eps0 = np.clip(ld[0] / b_air, 0.0, 1.0)
    try:
        pos3 = read_exr3(attempt / "frame_truth_world_position.exr",
                         ("X", "Y", "Z")).reshape(-1, 3)
        nrm3 = read_exr3(attempt / "frame_truth_geometric_normal.exr",
                         ("Nx", "Ny", "Nz")).reshape(-1, 3)
        cam = np.asarray(cfg["camera"]["position"], dtype=float)
        view = pos3 - cam[None, :]
        view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-12
        nn = nrm3 / (np.linalg.norm(nrm3, axis=1, keepdims=True) + 1e-12)
        refl = view - 2.0 * (view * nn).sum(axis=1, keepdims=True) * nn
        rl = np.linalg.norm(refl, axis=1) + 1e-12
        cos_z = refl[:, 1] / rl                       # world +Y is zenith
        sec = np.minimum(1.0 / np.maximum(np.abs(cos_z), 0.01), 5.0)
        l_dir = b_air[None, :] * (
            1.0 - (1.0 - eps0[None, :]) ** sec[:, None])
        sky_hit = cos_z > 0.0
        l_pred_dir = np.where(
            sky_hit[:, None],
            tau * (eps * bb + (1.0 - eps) * l_dir) + lp,
            l_pred)
        t_pred_dir = np.array([invert(v) for v in
                               (l_pred_dir * lam_w[None, :]).sum(axis=1)])
    except FileNotFoundError:
        t_pred_dir, sky_hit = None, None

    opaque_px = (valid.ravel() == 1) & (truth.ravel() > 0) & (~opaque)
    print(f"pixels: {n_px}  evaluated (valid && truth>0 && !opaque-path): "
          f"{opaque_px.sum()}  tau-opaque: {opaque.sum()}")
    res = t_pred - tapp.ravel()
    res_dir = (t_pred_dir - tapp.ravel()) if t_pred_dir is not None else res
    print(f"closure residual tapp_pred - tapp_meas [K]: "
          f"mean={res[opaque_px].mean():+.3f} rms={np.sqrt((res[opaque_px]**2).mean()):.3f} "
          f"p5={np.percentile(res[opaque_px],5):+.3f} p95={np.percentile(res[opaque_px],95):+.3f}")
    for mid in sorted(set(mat.ravel()[opaque_px].astype(int))):
        m = opaque_px & (mat.ravel() == mid)
        if m.sum() < 3:
            continue
        name = mats.get(mid, {}).get("name", "?")
        line = (f"  {name:<28} n={m.sum():4d}  bias={res[m].mean():+7.3f} K  "
                f"rms={np.sqrt((res[m]**2).mean()):7.3f} K")
        if t_pred_dir is not None and (m & sky_hit).sum() >= 3:
            ms = m & sky_hit
            line += (f"  | specular-sky n={ms.sum():4d} "
                     f"bias={res_dir[ms].mean():+7.3f} K  "
                     f"rms={np.sqrt((res_dir[ms]**2).mean()):7.3f} K")
        print(line)


if __name__ == "__main__":
    main()
