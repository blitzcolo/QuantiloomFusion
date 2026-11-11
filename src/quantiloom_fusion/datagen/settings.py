"""Validation of the dataset generation settings TOML."""

import math
import re
import tomllib

from ..core.paths import canonical_path, native_path
from ..core.vec import add, norm, scale
from .constants import BANDS


def check_keys(table, allowed, name):
    unknown = set(table) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {name} parameter(s): {sorted(unknown)}")


def positive(value, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def vector(v, name):
    if not isinstance(v, list) or len(v) != 3 or any(isinstance(x, bool) or not isinstance(x, (int,float)) or not math.isfinite(x) for x in v):
        raise ValueError(f"{name} requires three finite numbers")
    return [float(x) for x in v]


def read_settings(path):
    doc=tomllib.loads(path.read_text(encoding="utf-8-sig"))
    check_keys(doc,{"dataset","sampling","scenes","cameras"},"root")
    d=doc["dataset"]
    check_keys(d,{"name","qt_root","cli","seed","spp","wavelength_samples","thermal_mode","atmosphere_mode","atmosphere_model_pack","solar_ir_policy","daylight_solar_scale","ir_materials","noise_seed_policy","geometry_truth","verification","closure_audit_every"},"dataset")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+",d["name"]):raise ValueError("unsafe dataset name")
    for k in ("spp","wavelength_samples"):positive(d[k],k,True)
    if type(d["seed"]) is not int or not 0<=d["seed"]<=0xffffffff:raise ValueError("seed must be uint32")
    if d["thermal_mode"] not in ("authored","solved"):raise ValueError("thermal_mode must be authored or solved")
    if type(d.get("closure_audit_every",0)) is not int or d.get("closure_audit_every",0)<0:
        raise ValueError("closure_audit_every must be a nonnegative integer")
    if d["atmosphere_mode"] not in ("source","off"):raise ValueError("atmosphere_mode must be source or off")
    if d["solar_ir_policy"] not in ("planck_tail","source"):raise ValueError("solar_ir_policy must be planck_tail or source")
    if d.get("daylight_solar_scale") not in ("forcing_direct","none"):raise ValueError("daylight_solar_scale must be forcing_direct or none")
    if d["ir_materials"] not in ("constant","spectral"):raise ValueError("ir_materials must be constant or spectral")
    if d["noise_seed_policy"] not in ("fixed_detector","per_sample","per_acquisition"):raise ValueError("noise_seed_policy must be fixed_detector, per_sample or per_acquisition")
    if "atmosphere_model_pack" in d:
        p=d["atmosphere_model_pack"]
        if not isinstance(p,str) or not p:raise ValueError("atmosphere_model_pack must be a path string")
        d["atmosphere_model_pack"]=canonical_path(native_path(p))
    if type(d["geometry_truth"]) is not bool:raise ValueError("geometry_truth must be boolean")
    if d["verification"] not in ("hashes","pixels"):raise ValueError("verification must be hashes or pixels")
    if set(doc["cameras"])!=set(BANDS):raise ValueError("exactly vis, mwir, lwir profiles are required")
    for band,c in doc["cameras"].items():
        check_keys(c,{"resolution","fov_y_deg","pixel_pitch_um","f_number","exposure_s","offset_rdf_m","rotation_ypr_deg","psf_sigma_px","read_noise_e_rms","prnu_sigma","dsnu_e_rms","full_well_e","electrons_per_dn","adc_bits","netd_k","responsivity_dn_w","time_constant_s","warmup_s","distortion","noise_free"},f"cameras.{band}")
        if len(c["resolution"])!=2:raise ValueError("resolution requires width,height")
        for x in c["resolution"]:positive(x,"resolution",True)
        for k in ("fov_y_deg","pixel_pitch_um","f_number","exposure_s"):positive(c[k],k)
        if c["fov_y_deg"]>=160:raise ValueError("fov_y_deg must be below 160")
        for k in ("offset_rdf_m","rotation_ypr_deg"):vector(c[k],k)
        if norm(c["offset_rdf_m"])>1:raise ValueError("camera offsets must be small (<=1 metre)")
        if len(c["distortion"])!=5 or any(not math.isfinite(x) for x in c["distortion"]):raise ValueError("Brown distortion requires five finite coefficients")
        for k in ("psf_sigma_px","read_noise_e_rms","prnu_sigma","dsnu_e_rms","netd_k","time_constant_s","warmup_s"):
            if k in c and (not math.isfinite(c[k]) or c[k]<0):raise ValueError(f"{k} must be nonnegative")
        if band=="lwir":positive(c["responsivity_dn_w"],"responsivity_dn_w")
        else:
            for k in ("full_well_e","electrons_per_dn"):positive(c[k],k)
        if type(c['noise_free']) is not bool:raise ValueError('noise_free must be boolean')
        if type(c['adc_bits']) is not int or not 1<=c["adc_bits"]<=24:raise ValueError("adc_bits must be integer in 1..24")
    # A coincident rig would violate this generator's experimental contract.
    offsets=[doc["cameras"][b]["offset_rdf_m"] for b in BANDS]
    if any(norm(add(offsets[i],scale(offsets[j],-1)))<1e-5 for i in range(3) for j in range(i)):
        raise ValueError("three cameras require distinct local positions")
    s=doc["sampling"]
    check_keys(s,{"azimuths_deg","elevations_deg","radius_scales","repeats"},"sampling")
    for key in ("azimuths_deg","elevations_deg","radius_scales"):
        if not s[key] or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in s[key]):raise ValueError(f"invalid {key}")
        if len(s[key])!=len(set(s[key])):raise ValueError(f"duplicate {key}")
    if any(not 0<=a<360 for a in s["azimuths_deg"]):raise ValueError("azimuth must be in [0,360)")
    if any(not -80<e<80 for e in s["elevations_deg"]):raise ValueError("elevation must be in (-80,80)")
    for v in s["radius_scales"]:positive(v,"radius_scale")
    positive(s["repeats"],"repeats",True)
    names=[]
    for scene in doc["scenes"]:
        check_keys(scene,{"id","config","target","radius_m","overrides","solar_scale","vis_exposure_s"},"scene")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+",scene["id"]):raise ValueError("unsafe scene id")
        if "overrides" in scene and not isinstance(scene["overrides"],dict):
            raise ValueError("scene overrides must be a table of named sections to replace")
        if "vis_exposure_s" in scene:positive(scene["vis_exposure_s"],"vis_exposure_s")
        names.append(scene["id"])
        if "target" in scene:vector(scene["target"],"target")
        if "radius_m" in scene:positive(scene["radius_m"],"radius_m")
    if len(names)!=len(set(names)) or not names:raise ValueError("duplicate or missing scenes")
    # Record paths canonically so a dataset planned on one OS resumes on the other.
    d["qt_root"]=canonical_path(native_path(d["qt_root"]).resolve())
    d["cli"]=canonical_path(native_path(d["cli"]).resolve())
    return doc
