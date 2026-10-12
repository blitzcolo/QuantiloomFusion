"""Quantify atmospheric spectral-integration error for the camera lambda grid.

Compares three ways of reducing the NN channel function to a band number:

  exact     - channel-quantized function integrated over its native cells.
              Each wavenumber channel owns the lambda-interval between its
              wavenumber midpoints; the integral is sum(v_c * cell_width).
  nearest   - trapezoid over the camera grid, each sample taking the nearest
              wavenumber channel (the renderer's old single-lambda path).
  boxed     - trapezoid over the camera grid, each sample taking the mean of
              the channels inside its Voronoi cell (the new bake).

Usage: _spectral_integration_audit.py [band] [model_pack]
band defaults to "all"; model_pack defaults to the Quantiloom-dev atmosphere
model pack (H:/Quantiloom-dev/assets/atmos_models, translated on WSL).
Prints band-mean values and relative errors per scheme.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from _closure_audit import Net, fill_row, planck_nm  # noqa: E402
from quantiloom_fusion.core.paths import native_path  # noqa: E402

# Night-arm weather (the 23:00 forcing values used by the smoke).
WEATHER = dict(atmos_model=2.0, ihaze=1.0, icld=0.0, vis_km=23.0,
               rainrt_mm_h=0.0, t_ground_K=300.0, rh=48.0,
               p_hPa=1013.25, h2o_scale=1.0)
SUN_ZEN = 85.0  # networks are clamped to the day branch at 85 deg
H1_KM = 0.002
RANGE_KM = 0.1


def channel_cells_nm(net):
    """Each channel's lambda-interval from wavenumber midpoints."""
    nu = net.wavenumbers()
    edges = np.concatenate([[nu[0] - (nu[1] - nu[0]) / 2],
                            (nu[:-1] + nu[1:]) / 2,
                            [nu[-1] + (nu[1] - nu[0]) / 2]])
    lam_hi = 1e7 / edges[:-1]   # lower nu -> higher lambda
    lam_lo = 1e7 / edges[1:]
    lam_c = 1e7 / nu
    return lam_lo, lam_c, lam_hi


def trapz_weights(lam):
    w = np.empty(len(lam))
    w[0], w[-1] = (lam[1] - lam[0]) / 2, (lam[-1] - lam[-2]) / 2
    w[1:-1] = (lam[2:] - lam[:-2]) / 2
    return w


def voronoi_cells(lam):
    mids = (lam[:-1] + lam[1:]) / 2
    lo = np.concatenate([[lam[0]], mids])
    hi = np.concatenate([mids, [lam[-1]]])
    return lo, hi


def evaluate(net, is_delta, band_lo_nm, band_hi_nm, samples, weight):
    """Return (exact, nearest, boxed) band-mean of the channel field."""
    lam_lo, lam_c, lam_hi = channel_cells_nm(net)
    row = fill_row(net, WEATHER, SUN_ZEN, 0.0, H1_KM, 90.0, RANGE_KM)[None, :]
    v = net.forward(row)[0]
    k = len(lam_c)
    if len(v) > k:
        v = v.reshape(len(v) // k, k)[0]  # multi-target net: first component
    v = np.exp(-v) if is_delta else v * 1e11 / (lam_c ** 2)  # cm -> nm

    lam = np.linspace(band_lo_nm, band_hi_nm, samples + 1)
    w = trapz_weights(lam)
    clo, chi = voronoi_cells(lam)
    W_exact = weight(lam_c)

    lo = np.maximum(lam_lo, band_lo_nm)
    hi = np.minimum(lam_hi, band_hi_nm)
    cw = np.clip(hi - lo, 0.0, None)
    exact = float((v * W_exact * cw).sum() / (W_exact * cw).sum())

    Wl = weight(lam)
    denom = float((Wl * w).sum())
    idx = np.argmin(np.abs(lam_c[None, :] - lam[:, None]), axis=1)
    nearest = float((v[idx] * Wl * w).sum() / denom)
    boxed_v = []
    for i in range(len(lam)):
        m = (lam_c >= clo[i]) & (lam_c <= chi[i])
        boxed_v.append(v[m].mean() if m.any() else v[idx[i]])
    boxed = float((np.asarray(boxed_v) * Wl * w).sum() / denom)
    return exact, nearest, boxed


def main():
    band = sys.argv[1] if len(sys.argv) > 1 else "all"
    models = native_path(sys.argv[2]) if len(sys.argv) > 2 else native_path("H:/Quantiloom-dev/assets/atmos_models")
    planck300 = lambda l: planck_nm(np.asarray(l, dtype=float), 300.0)
    cases = {
        "lwir": dict(band=(8000.0, 12000.0),
                     nets={c: (models / f"lwir_ground_{c}.safetensors",
                               c == "tau")
                           for c in ("tau", "lpath", "ldown")}),
        "mwir": dict(band=(3000.0, 5000.0),
                     nets={c: (models / f"mwir_ground_{c}.safetensors",
                               c == "tau")
                           for c in ("tau", "lpath", "ldown")}),
    }
    for band_name in (cases if band == "all" else [band]):
        cfg = cases[band_name]
        for c, (p, is_delta) in cfg["nets"].items():
            if not Path(p).is_file():
                print(f"{band_name} {c}: {p} missing, skipped")
                continue
            net = Net(p)
            for samples in (4, 12, 24, 48):
                exact, nearest, boxed = evaluate(
                    net, is_delta, *cfg["band"], samples, planck300)
                en = 100 * (nearest - exact) / exact
                eb = 100 * (boxed - exact) / exact
                print(f"{band_name} {c:5s} N={samples + 1:2d}  "
                      f"exact={exact:.6g}  nearest={nearest:.6g} ({en:+.2f}%)  "
                      f"boxed={boxed:.6g} ({eb:+.3f}%)")


if __name__ == "__main__":
    main()
