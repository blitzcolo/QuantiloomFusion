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


# Named presets, as AtmosphereNNConfig.cpp resolves them before explicit keys
# override: (atmos_model, ihaze, icld, vis_km, rainrt_mm_h, t_ground_K, rh, p_hPa).
PRESETS = {
    "clear": (2, 1, 0, 23.0, 0.0, 293.15, 0.50, 1013.25),
    "turbulent_clear": (2, 1, 0, 20.0, 0.0, 308.15, 0.30, 1010.00),
    "urban_haze": (2, 5, 0, 8.0, 0.0, 298.15, 0.75, 1010.00),
    "fog": (2, 9, 0, 0.5, 0.0, 288.15, 1.00, 1015.00),
    "light_rain": (2, 4, 6, 10.0, 2.5, 290.15, 0.85, 1008.00),
    "heavy_rain": (2, 4, 6, 2.0, 25.0, 288.15, 0.95, 1005.00),
    "snow": (3, 1, 6, 2.5, 5.0, 273.15, 0.90, 1020.00),
    "haze": (2, 5, 0, 3.0, 0.0, 303.15, 0.40, 1000.00),
}
WEATHER_KEYS = ("atmos_model", "ihaze", "icld", "vis_km", "rainrt_mm_h", "t_ground_K", "rh", "p_hPa")


def resolve_weather(atmos):
    base = PRESETS.get(atmos.get("preset", "clear"), PRESETS["clear"])
    w = {k: float(atmos.get(k, v)) for k, v in zip(WEATHER_KEYS, base)}
    if w["rh"] > 1.0:  # the renderer reads rh > 1 as a percentage
        w["rh"] /= 100.0
    w["h2o_scale"] = float(atmos.get("h2o_scale", 1.0))
    return w


def read_nk(path):
    """(lambda_um, n, k) rows of a refractiveindex.info 'tabulated nk' YAML."""
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            rows.append([float(v) for v in parts])
        except ValueError:
            continue
    if len(rows) < 2:
        raise ValueError(f"{path}: no tabulated n,k rows")
    a = np.asarray(sorted(rows))
    return a[:, 0], a[:, 1], a[:, 2]


def fresnel_unpolarised(cos_i, n, k):
    """Unpolarised Fresnel reflectance of air -> n + ik at incidence cos_i.

    cos_i: (P,), n, k: (L,) -> (P, L)."""
    N = (n + 1j * k)[None, :]
    ci = np.clip(cos_i, 1e-6, 1.0)[:, None]
    st2 = 1.0 - ci**2
    ct = np.sqrt(1.0 - st2 / N**2)
    rs = (ci - N * ct) / (ci + N * ct)
    rp = (N * ci - ct) / (N * ci + ct)
    return 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)


def slab_sky(b_air, eps0, cos_zenith):
    """The renderer's flat-slab sky (AtmosSkyRadianceIR): B(T_air)[1-(1-eps0)^sec]."""
    sec = np.minimum(1.0 / np.maximum(cos_zenith, 0.01), 5.0)
    return b_air[None, :] * (1.0 - (1.0 - eps0[None, :]) ** sec[:, None])


def invert_many(l_spec, lam_grid, lam_w):
    """Band-average radiance -> apparent temperature, vectorised bisection."""
    target = (l_spec * lam_w[None, :]).sum(axis=1)
    lo = np.full(target.shape, 10.0)
    hi = np.full(target.shape, 5000.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        with np.errstate(over="ignore"):  # cold bisection bounds: B -> 0 exactly as expm1 -> inf
            v = (planck_nm(lam_grid[None, :], mid[:, None]) * lam_w[None, :]).sum(axis=1)
        below = v < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def stats(r, m):
    return f"n={m.sum():5d} bias={r[m].mean():+7.3f} K rms={np.sqrt((r[m] ** 2).mean()):7.3f} K"


def main():
    attempt = Path(sys.argv[1])
    cfg = tomllib.loads((attempt / "capture.toml").read_text())
    if "model_pack" not in cfg["atmosphere"]:
        sys.exit("capture.toml has no atmosphere.model_pack")
    models = native_path(cfg["atmosphere"]["model_pack"])
    atmos = cfg["atmosphere"]
    weather = resolve_weather(atmos)
    sun_dir = np.asarray(cfg["lighting"]["sun_direction"], dtype=float)
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    sun_zen = math.degrees(math.acos(np.clip(sun_dir[1], -1.0, 1.0)))
    h1_km = float(atmos.get("h1_km", 0.1))
    w2m = float(cfg["scene"].get("world_units_to_meters", 1.0))
    sun_zen_bake = min(sun_zen, 85.0)

    tapp = read_exr(attempt / "frame_tapp.exr").ravel()
    truth = read_exr(attempt / "frame_truth_surface_temperature.exr").ravel()
    valid = read_exr(attempt / "frame_truth_validity.exr").ravel()
    flat = read_exr(attempt / "frame_truth_material_id.exr").astype(int).ravel()
    dist = read_exr(attempt / "frame_truth_ray_distance.exr").ravel()
    mats = {m["material_id"]: m for m in
            json.loads((attempt / "frame_truth_materials.json").read_text())}

    lam_grid = np.array([8000.0, 9000.0, 10000.0, 11000.0, 12000.0])
    lam_w = np.array([0.5, 1.0, 1.0, 1.0, 0.5]) * 1000.0 / 4000.0  # trapezoid

    nets = {c: Net(models / f"lwir_ground_{c}.safetensors")
            for c in ("tau", "lpath", "ldown")}

    # Each net once per pixel at its own slant range (the baker's ground path
    # is horizontal: view zenith 90 deg, path range feature).
    ranges_km = np.clip(dist * w2m / 1000.0, 0.05, 20.0)
    n_px = ranges_km.size

    def infer_at_channels(net, rows):
        out = net.forward(rows)
        return out[:, [net.channel_at(l) for l in lam_grid]]

    def nm_scale(net):
        lam = 1e7 / net.wavenumbers()
        ch = [net.channel_at(l) for l in lam_grid]
        return 1e11 / (lam[ch] ** 2)

    tau_rows = np.stack([fill_row(nets["tau"], weather, sun_zen_bake, 0.0, h1_km, 90.0, r)
                         for r in ranges_km])
    lpath_rows = np.stack([fill_row(nets["lpath"], weather, sun_zen_bake, 0.0, h1_km, 90.0, r)
                           for r in ranges_km])
    ldown_row = fill_row(nets["ldown"], weather, sun_zen_bake, 0.0, h1_km, 90.0, 1.0)[None, :]
    delta = infer_at_channels(nets["tau"], tau_rows)
    opaque = delta.min(axis=1) > nets["tau"].opaque_delta
    tau = np.exp(-delta)
    # LWIR lpath is pure thermal emission: never scaled by solar_scatter_scale.
    lp = infer_at_channels(nets["lpath"], lpath_rows) * nm_scale(nets["lpath"])
    ld_zenith = (infer_at_channels(nets["ldown"], ldown_row) * nm_scale(nets["ldown"]))[0]

    # With the NN atmosphere active the renderer replaces the thermal-sky air
    # temperature by atmosphere.t_ground_K (ConfigResolve), whatever
    # lighting.atmosphere_temperature_k says; the slab law takes its zenith
    # emissivity from the baked ldown against B(T_air).
    t_air = weather["t_ground_K"]
    b_air = planck_nm(lam_grid, t_air)
    eps0 = np.clip(ld_zenith / b_air, 0.0, 1.0)
    # Cosine-weighted hemispherical mean of the slab sky: what an unoccluded
    # horizontal Lambertian surface reflects.
    mu = (np.arange(400) + 0.5) / 400.0
    ld_hemi = 2.0 * (slab_sky(b_air, eps0, mu) * mu[:, None]).mean(axis=0)

    # Materials outside the thermal solve export no surface temperature (0);
    # the renderer emits them at their authored temperature, so use that.
    authored = np.array([mats.get(m, {}).get("authored_ir_temperature_k") or 0.0 for m in flat])
    solved = np.array([bool(mats.get(m, {}).get("thermal_participates")) for m in flat])
    t_surface = np.where((truth <= 0) & ~solved & (authored > 0), authored, truth)
    with np.errstate(over="ignore"):
        bb = planck_nm(lam_grid[None, :], np.clip(t_surface, 1.0, 5000.0)[:, None])

    # Specular: the renderer lists n,k-bound surfaces as Fresnel with a null
    # band emissivity (angle dependent); since 0.10.0 they render a directional
    # Fresnel lobe in the infrared. Diffuse: measured spectrum or authored
    # emissivity, one band number.
    nk_tables = {k: v for k, v in (cfg.get("refractive_index") or {}).items() if isinstance(v, str)}
    specular_ids = {mid for mid, m in mats.items()
                    if m.get("emissivity_model") == "fresnel_complex_refractive_index"}
    diffuse_ids = {mid for mid, m in mats.items()
                   if mid not in specular_ids and m.get("band_emissivity") is not None}
    unknown_ids = set(mats) - specular_ids - diffuse_ids
    base = (valid == 1) & (truth > 0) & (~opaque)
    print(f"pixels: {n_px}  evaluated (valid && truth>0 && !opaque-path): {base.sum()}  "
          f"tau-opaque: {opaque.sum()}  T_air = t_ground_K = {t_air:.2f} K")
    for mid in sorted(specular_ids):
        if mats[mid]["name"] not in nk_tables:
            print(f"WARNING: {mats[mid]['name']} is Fresnel in the truth table but capture.toml "
                  f"has no [refractive_index] entry for it")
    for name in nk_tables:
        if not any(mats[mid]["name"] == name for mid in specular_ids):
            print(f"NOTE: [refractive_index] {name} is not listed as Fresnel in the truth table")

    eps_d = np.array([mats.get(m, {}).get("band_emissivity") or 0.0 for m in flat])
    eps_d = np.clip(eps_d, 0.0, 1.0)[:, None]
    diffuse_px = base & np.isin(flat, sorted(diffuse_ids))
    res_hemi = invert_many(tau * (eps_d * bb + (1.0 - eps_d) * ld_hemi[None, :]) + lp,
                           lam_grid, lam_w) - tapp
    res_zen = invert_many(tau * (eps_d * bb + (1.0 - eps_d) * ld_zenith[None, :]) + lp,
                          lam_grid, lam_w) - tapp
    if diffuse_px.any():
        r = res_hemi[diffuse_px]
        print(f"closure residual tapp_pred - tapp_meas [K]: mean={r.mean():+.3f} "
              f"rms={np.sqrt((r**2).mean()):.3f} p5={np.percentile(r,5):+.3f} "
              f"p95={np.percentile(r,95):+.3f} (diffuse materials, isotropic, hemispherical sky)")
    print("DIFFUSE materials -- tau [eps B(T_s) + (1 - eps) L_down] + L_path with the band "
          "emissivity of the truth table (one number per material, so emissivity multiplies "
          "after the band integral: the residual includes that band-average approximation). "
          "L_down = cosine-weighted hemispherical slab sky | zenith NN ldown for comparison")
    for mid in sorted(diffuse_ids):
        m = diffuse_px & (flat == mid)
        if m.sum() >= 3:
            print(f"  {mats[mid]['name']:<28} eps={mats[mid]['band_emissivity']:.3f} "
                  f"{stats(res_hemi, m)} | zenith {stats(res_zen, m)}")

    try:
        pos3 = read_exr3(attempt / "frame_truth_world_position.exr", ("X", "Y", "Z")).reshape(-1, 3)
        nrm3 = read_exr3(attempt / "frame_truth_geometric_normal.exr", ("Nx", "Ny", "Nz")).reshape(-1, 3)
    except FileNotFoundError:
        print("SPECULAR materials: no world-position/normal truth, not audited")
        return
    cam = np.asarray(cfg["camera"]["position"], dtype=float)
    view = pos3 - cam[None, :]
    view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-12
    nn = nrm3 / (np.linalg.norm(nrm3, axis=1, keepdims=True) + 1e-12)
    nn[(view * nn).sum(axis=1) > 0.0] *= -1.0  # normals face the camera
    cos_i = -(view * nn).sum(axis=1)
    refl = view + 2.0 * cos_i[:, None] * nn
    refl /= np.linalg.norm(refl, axis=1, keepdims=True) + 1e-12
    sky_dir = refl[:, 1] > 0.0  # world +Y is the zenith
    print("SPECULAR materials -- tau [(1 - F) B(T_s) + F L_sky(mirror)] + L_path, F = unpolarised "
          "Fresnel at the pixel's incidence from the bound n,k; only pixels whose mirror direction "
          "points at the sky (occlusion of that direction by scene geometry is not tested) | "
          "isotropic eps = 1 - F(normal) with zenith L_down for comparison")
    for mid in sorted(specular_ids):
        name = mats[mid]["name"]
        m_all = (valid == 1) & (t_surface > 0) & (~opaque) & (flat == mid)
        if name not in nk_tables or m_all.sum() < 3:
            continue
        lam_um, n_tab, k_tab = read_nk(native_path(nk_tables[name]))
        n = np.interp(lam_grid / 1000.0, lam_um, n_tab)
        k = np.interp(lam_grid / 1000.0, lam_um, k_tab)
        idx = np.nonzero(m_all & sky_dir)[0]
        if idx.size < 3:
            print(f"  {name:<28} n_all={m_all.sum():5d} sky-pointing={idx.size} (too few)")
            continue
        F = fresnel_unpolarised(cos_i[idx], n, k)
        l_spec = tau[idx] * ((1.0 - F) * bb[idx] + F * slab_sky(b_air, eps0, refl[idx, 1])) + lp[idx]
        r_spec = invert_many(l_spec, lam_grid, lam_w) - tapp[idx]
        F0 = fresnel_unpolarised(np.ones(1), n, k)[0]
        l_iso = tau[idx] * ((1.0 - F0)[None, :] * bb[idx] + F0[None, :] * ld_zenith[None, :]) + lp[idx]
        r_iso = invert_many(l_iso, lam_grid, lam_w) - tapp[idx]
        every = np.ones(idx.size, bool)
        interior = cos_i[idx] > 0.5  # away from the limb, where the PSF mixes in the background
        t_src = "solved" if mats[mid].get("thermal_participates") else f"authored {mats[mid].get('authored_ir_temperature_k')} K"
        line = (f"  {name:<28} T_s={t_src} n_all={m_all.sum():5d} excluded(mirror below horizon)={m_all.sum() - idx.size:5d} "
                f"eps(normal)={1.0 - (F0 * lam_w).sum():.3f} | specular {stats(r_spec, every)} "
                f"| isotropic {stats(r_iso, every)}")
        if interior.sum() >= 3:
            line += f" | interior cos_i>0.5: specular {stats(r_spec, interior)}, isotropic {stats(r_iso, interior)}"
        print(line)
    for mid in sorted(unknown_ids):
        m = base & (flat == mid)
        if m.sum() >= 3:
            print(f"  (not audited) {mats[mid]['name']:<28} n={m.sum():5d} "
                  f"emissivity_model={mats[mid].get('emissivity_model')}")


if __name__ == "__main__":
    main()
