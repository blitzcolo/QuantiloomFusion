#!/usr/bin/env python3
"""Plan, render, resume and verify static, unregistered three-band datasets.

Python 3.11+; planning and hash verification require only the standard library.
No renderer/SDK imports. Optional pixel QA uses NumPy and OpenEXR.

Runs under Windows Python or a WSL2 Linux Python. The renderer is always the
Windows CLI; on WSL it is invoked via interop, so the dataset must live on a
Windows-accessible drive (under /mnt/<drive>/). Recipe identity is gated by the
CLI `--version` string plus content hashes of the scene assets, settings and
this script; renderer binaries are recorded for provenance but do not gate.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import sys
import tomllib
import uuid
from itertools import product

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantiloom_fusion.core.hashing import digest, fingerprint
from quantiloom_fusion.core.host import POSIX
from quantiloom_fusion.core.jsonio import atomic_json, load_json
from quantiloom_fusion.core.locking import alive, runner_lock
from quantiloom_fusion.core.paths import canonical_path, is_drive_path, native_path
from quantiloom_fusion.core.tomlio import scalar, toml_text
from quantiloom_fusion.core.vec import add, camera_pose, cross, dot, local_rotation, norm, pose, scale, unit

SCHEMA = "quantiloomfusion.env_dataset"
VERSION = 1
BANDS = ("vis", "mwir", "lwir")
MODES = {"vis": "vis_hero", "mwir": "mwir_fused", "lwir": "lwir_fused"}
PATH_KEYS = {"gltf", "usd", "file", "basis_file", "materials_json", "solar_lut",
             "forcing_file", "model_pack", "environment_map", "temperature_texture",
             "known_psf_path", "emissive_curve", "spectral_weight_texture"}


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


def reject_motion(doc):
    if "timeline" in doc or "motion" in doc.get("camera", {}):
        raise ValueError("only static scene snapshots are supported")
    for model in doc.get("models", []):
        if "motion" in model:
            raise ValueError("model motion is forbidden, including nested motion tables")


def read_gltf(path):
    if path.suffix.lower()==".glb":
        with path.open("rb") as f:
            magic,version,size=struct.unpack("<III",f.read(12))
            if magic!=0x46546c67 or version!=2:raise ValueError("invalid GLB")
            length,kind=struct.unpack("<II",f.read(8))
            if kind!=0x4e4f534a:raise ValueError("GLB has no initial JSON chunk")
            doc=json.loads(f.read(length))
    else:doc=load_json(path)
    if doc.get("animations"):
        raise ValueError("animated glTF must be explicitly frozen before dataset generation")
    return doc


def resolve_asset(value, config_dir, qt_root, cli_root):
    p=native_path(value)
    candidates=[p] if p.is_absolute() else [config_dir/p,qt_root/p,cli_root/p]
    for candidate in candidates:
        if candidate.exists():return candidate.resolve()
    raise FileNotFoundError(f"asset not found: {value}; searched {candidates}")


def canonicalize(doc, config_dir, qt_root, cli_root):
    refs=set()
    def visit(table, parent=""):
        if isinstance(table,list):
            for entry in table:
                if isinstance(entry,dict):visit(entry,parent)
        elif isinstance(table,dict):
            for key,value in table.items():
                if isinstance(value,(dict,list)):visit(value,key)
                elif isinstance(value,str) and value and (key in PATH_KEYS or parent in ("refractive_index","spectral_curves")):
                    p=resolve_asset(value,config_dir,qt_root,cli_root)
                    table[key]=canonical_path(p);refs.add(p)
    visit(doc)
    # Qt's environment editor often relies on the installed default NN pack.
    if "atmosphere" in doc and "model_pack" not in doc["atmosphere"]:
        pack=qt_root/'build-msvc/Release/assets/atmos_models'
        if not pack.exists():pack=cli_root/'assets/atmos_models'
        if not pack.exists():raise FileNotFoundError("source atmosphere requires an explicit model_pack")
        doc["atmosphere"]["model_pack"]=canonical_path(pack.resolve());refs.add(pack.resolve())
    return refs


def solar_extension(source, destination, scale=1.0):
    # An explicitly labelled modelling assumption, not measured IR sunlight.
    with source.open(encoding="utf-8-sig") as f:
        rows=list(csv.reader(f))
    header=rows[0];values=[[float(v) for v in row] for row in rows[1:] if row]
    last=values[-1];anchor=last[0]
    def ratio(w):
        c2=1.438776877e7  # nm K
        return (anchor/w)**5*math.expm1(c2/(anchor*5778))/math.expm1(c2/(w*5778))
    if anchor<15000:
        for w in range(int(anchor)+100,15001,100):
            # Fixed formatting keeps the derived table byte-identical across OS libm.
            values.append([float(w),*[f'{v*ratio(w):.12e}' for v in last[1:]]])
    if scale!=1.0:
        # Low-light arm: every irradiance column is scaled together so the
        # direct/diffuse split the resolver derives is preserved. Tail rows
        # were already formatted, so the value goes back through float.
        values=[[row[0],*[f'{float(v)*scale:.12e}' for v in row[1:]]] for row in values]
    buffer=io.StringIO(newline='')
    writer=csv.writer(buffer);writer.writerow(header);writer.writerows(values)
    return buffer.getvalue()


def read_settings(path):
    doc=tomllib.loads(path.read_text(encoding="utf-8-sig"))
    check_keys(doc,{"dataset","sampling","scenes","cameras"},"root")
    d=doc["dataset"]
    check_keys(d,{"name","qt_root","cli","seed","spp","wavelength_samples","thermal_mode","atmosphere_mode","atmosphere_model_pack","solar_ir_policy","ir_materials","noise_seed_policy","geometry_truth","verification","closure_audit_every"},"dataset")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+",d["name"]):raise ValueError("unsafe dataset name")
    for k in ("spp","wavelength_samples"):positive(d[k],k,True)
    if type(d["seed"]) is not int or not 0<=d["seed"]<=0xffffffff:raise ValueError("seed must be uint32")
    if d["thermal_mode"] not in ("authored","solved"):raise ValueError("thermal_mode must be authored or solved")
    if type(d.get("closure_audit_every",0)) is not int or d.get("closure_audit_every",0)<0:
        raise ValueError("closure_audit_every must be a nonnegative integer")
    if d["atmosphere_mode"] not in ("source","off"):raise ValueError("atmosphere_mode must be source or off")
    if d["solar_ir_policy"] not in ("planck_tail","source"):raise ValueError("solar_ir_policy must be planck_tail or source")
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


def sensor_config(band,c,d,identity):
    w,h=c["resolution"]
    focal=(h*c["pixel_pitch_um"]*1e-3)/(2*math.tan(math.radians(c["fov_y_deg"])/2))
    # Three policies. per_acquisition keeps one detector seed per band -- the
    # FPN classes ignore the acquisition counter by shader design, so PRNU/DSNU
    # stay put across views -- while dataset.acquisition_index shifts the
    # counter the temporal classes key on, so each capture gets its own
    # temporal realisation. per_sample reseeds the detector per capture (all
    # classes change, FPN included). fixed_detector is the legacy stream.
    key=identity if d["noise_seed_policy"]=="per_sample" else "fixed_detector"
    sensor={"version":1,"enabled":True,"input_kind":"spectral",
        "noise_seed":int(fingerprint([d['seed'],band,key])[:8],16),
        "preset":{"vis":"generic_cmos","mwir":"generic_cooled_photon_ir","lwir":"generic_uncooled_thermal_ir"}[band],
        "optics":{"focal_length_mm":focal,"pixel_pitch_um":c["pixel_pitch_um"],"f_number":c["f_number"],"sensor_width_px":w,"sensor_height_px":h,"psf_sigma_px":c["psf_sigma_px"],
                  "projection":{"model":"brown_conrady","coefficients":c["distortion"]}},
        "exposure":{"time_s":c["exposure_s"]},
        "readout":{"shutter":"global","frame_period_s":max(1/30,c['exposure_s']),"analog_gain":1.0,"adc_bits":c["adc_bits"],"output_bits":c["adc_bits"]},
        "quality":{"noise_free":c["noise_free"],"wavelength_samples":d["wavelength_samples"],"time_samples":1,"pixel_samples":1},
        "products":{"traced_radiance":False,"cie_linear_srgb":False,"band_measurement":True,"raw_dn":True,"corrected_device_signal":True,"apparent_temperature":band!="vis","display":True},
        "warmup":{"seconds":c["warmup_s"]}}
    if band=="lwir":
        sensor.update(effective_min_nm=8000.0,effective_max_nm=12000.0)
        sensor["channels"]=[{"name":"Mono","absorptance":{"kind":"thermal_absorptance","wavelength_nm":[8000.0,12000.0],"value":[.8,.8]}}]
        sensor["thermal"]={"netd_k":c["netd_k"],"responsivity_dn_w":c["responsivity_dn_w"],"time_constant_s":c["time_constant_s"]}
    else:
        sensor["photon"]={"full_well_e":c["full_well_e"],"electrons_per_dn":c["electrons_per_dn"],"read_noise_e_rms":c["read_noise_e_rms"],"prnu_sigma":c["prnu_sigma"],"dsnu_e_rms":c["dsnu_e_rms"],"enable_fpn":True}
        # electrons_per_dn is a readout field, never a photon parameter.
        sensor["readout"]["electrons_per_dn"]=sensor["photon"].pop("electrons_per_dn")
        if band=="vis":
            sensor["channels"]=[]
            for name,wave,val in [("R",[400.,540.,610.,700.],[0.,0.,.55,.25]),("G",[400.,470.,540.,620.,700.],[0.,.2,.6,.1,0.]),("B",[400.,460.,520.,700.],[.2,.55,0.,0.])]:
                sensor["channels"].append({"name":name,"qe":{"kind":"absolute_qe","wavelength_nm":wave,"value":val}})
    return sensor


def nk_span_nm(path):
    """(min_nm, max_nm) covered by a refractiveindex.info YAML table, else None.
    Tabulated files carry "<um> <n> <k>" rows; only lines whose first two
    columns are numeric count, so references and formula coefficients are
    skipped. Unreadable or unparseable files report no span."""
    try:text=Path(path).read_text(errors="replace")
    except(OSError,ValueError):return None
    values=[]
    for line in text.splitlines():
        parts=line.split()
        if not 2<=len(parts)<=3:continue
        try:row=[float(p) for p in parts]
        except ValueError:continue
        values.append(row[0])
    if len(values)<2:return None
    return min(values)*1000.0,max(values)*1000.0


def scene_for_band(base,band,materials,d,solar,acq=None):
    doc=copy.deepcopy(base)
    for key in ("sensor","isp","effects","postprocess","thermography","dataset"):
        doc.pop(key,None)
    doc.setdefault("renderer",{}).update(spp=d["spp"],environment_map_enabled=False)
    limits={'vis':(400,700),'mwir':(3000,5000),'lwir':(8000,12000)}[band]
    doc.setdefault("spectral",{}).update(mode=MODES[band],band=band.upper(),
        wavelength_nm=sum(limits)/2,lambda_min=limits[0],lambda_max=limits[1])
    doc["quality"]={"fail_on_srgb_upsample":False,"log_material_sources":True}
    doc["dataset"]={"metadata":True,"geometry_truth":d["geometry_truth"]}
    if acq is not None:doc["dataset"]["acquisition_index"]=acq
    if d["atmosphere_mode"]=="off":doc.pop("atmosphere",None)
    elif d.get("atmosphere_model_pack") and "atmosphere" in doc:
        # One pack for the whole dataset: the NN model is part of the recipe,
        # not something each scene may resolve differently. A scene with no
        # [atmosphere] section keeps having none -- a pack is not invented.
        doc["atmosphere"]["model_pack"]=d["atmosphere_model_pack"]
    if d["thermal_mode"]=="authored":doc.setdefault("thermal",{})["enabled"]=False
    if solar:doc.setdefault("lighting",{})["solar_lut"]=canonical_path(solar)
    entries={m["name"]:m for m in doc.get("materials",[])}
    overrides=doc.get("material_overrides",{})
    if band=="vis":
        # The solver only feeds emission, which is zero at visible
        # wavelengths; dropping [thermal] keeps the visible job byte-identical
        # across thermal_mode arms so ablations can share captures.
        doc.pop("thermal",None)
        # Legacy IR scalar fractions span IR wavelengths only; do not pretend
        # their generated curve is a measured visible reflectance.
        for m in [*entries.values(),*overrides.values()]:
            m.pop("ir_emissivity",None);m.pop("ir_transmittance",None)
    elif d["ir_materials"]=="spectral":
        # Bound materials render their measured spectrum in every band,
        # including IR; emissivity follows Kirchhoff from that reflectance, so
        # authored ir_emissivity/ir_transmittance must be dropped on them or the
        # resolver's energy check (rho+epsilon<=1) aborts the render. Unbound
        # materials keep the scene's constant emissivity.
        bound={m.get("name") for m in [*entries.values(),*overrides.values()]
               if m.get("spectral_material_ref") or m.get("spectral_material_refs")}
        for m in [*entries.values(),*overrides.values()]:
            if m.get("name") in bound:
                m.pop("ir_emissivity",None);m.pop("ir_transmittance",None)
        for material in materials:
            name=material["name"]
            m=entries.setdefault(name,{"name":name})
            if name not in bound:
                m.setdefault("ir_emissivity",overrides.get(name,{}).get("ir_emissivity",.9))
        # Complex refractive index files stay only where the source data covers
        # the band: a VIS-NIR nk table cannot drive LWIR optics and the resolver
        # treats non-covering data as fatal. Bound materials fall back to their
        # measured reflectance, which keeps real-world glass opaque in LWIR.
        ri=doc.get("refractive_index")
        if isinstance(ri,dict):
            for material_name,table in list(ri.items()):
                if not isinstance(table,str):continue
                span=nk_span_nm(native_path(table))
                if span is None or span[0]>limits[0] or span[1]<limits[1]:
                    del ri[material_name]
            if all(not isinstance(v,str) for v in ri.values()):doc.pop("refractive_index")
    else:
        # Qt env scenes bind USGS materials whose IR support is often absent.
        # Explicitly use the scene author's constant IR emissivity assumptions.
        for key in ("basis_file","materials_json"):doc["spectral"].pop(key,None)
        doc.pop("refractive_index",None);doc.pop("spectral_curves",None)
        for material in materials:
            name=material["name"]
            m=entries.setdefault(name,{"name":name})
            for key in list(m):
                if key.startswith("spectral_material_"):m.pop(key)
            m.setdefault("ir_emissivity",overrides.get(name,{}).get("ir_emissivity",.9))
        for m in overrides.values():
            for key in list(m):
                if key.startswith("spectral_material_"):m.pop(key)
    doc["materials"]=list(entries.values())
    return doc


def plan(config: Path,out: Path):
    settings=read_settings(config);d=settings["dataset"]
    qt=native_path(d["qt_root"]).resolve();cli=native_path(d["cli"]).resolve()
    if not cli.is_file():raise FileNotFoundError(f"CLI not found: {cli}")
    cli_root=next((p for p in cli.parents if (p/'assets').is_dir() and (p/'src').is_dir()),cli.parent)
    # Renderer binaries are observed, not gated: the CLI version string is the
    # compatibility contract, so a same-version rebuild must not force a new
    # dataset. Scene assets still gate by content hash.
    renderer_files={cli,*cli.parent.glob('*.spv'),*cli.parent.glob('*.dll')}
    if (cli.parent/'usd').is_dir():renderer_files.add(cli.parent/'usd')
    assets=set()
    samples=[];scene_info=[];texts={};derived={}
    out.mkdir(parents=True,exist_ok=True)
    if d.get("atmosphere_model_pack"):
        pack=native_path(d["atmosphere_model_pack"]).resolve()
        if not pack.is_dir():raise FileNotFoundError(f"atmosphere_model_pack not found: {pack}")
        assets.add(pack)
    for definition in settings["scenes"]:
        source=resolve_asset(definition["config"],qt/'assets/configs',qt,cli_root)
        assets.add(source)
        base=tomllib.loads(source.read_text(encoding="utf-8-sig"))
        # Scene overrides replace whole named sections (write the complete
        # [lighting]/[thermal]/... table in the settings, not a patch of it).
        for key,value in definition.get("overrides",{}).items():base[key]=value
        reject_motion(base)
        if d["thermal_mode"]=="solved" and not base.get("thermal",{}).get("enabled"):
            raise ValueError(f'thermal_mode="solved" but {definition["id"]} has no enabled [thermal] section')
        refs=canonicalize(base,source.parent,qt,cli_root);assets.update(refs)
        model=native_path(base['scene']['gltf']);gltf=read_gltf(model)
        model_hash=digest(model)
        for collection in ('buffers','images'):
            for resource in gltf.get(collection,[]):
                uri=resource.get('uri','')
                if uri and not uri.startswith('data:'):
                    p=resolve_asset(uri,model.parent,qt,cli_root);assets.add(p)
        ws=float(base['scene'].get('world_units_to_meters',1));positive(ws,'world_units_to_meters')
        target=definition.get('target',base['camera']['look_at'])
        radius=definition.get('radius_m',norm(add(base['camera']['position'],scale(target,-1)))*ws)
        materials=[m for m in gltf.get('materials',[]) if m.get('name')]
        solar=None
        solar_scale=definition.get('solar_scale',1.0)
        if not isinstance(solar_scale,(int,float)) or isinstance(solar_scale,bool) or not math.isfinite(solar_scale) or solar_scale<=0:
            raise ValueError(f'scene {definition["id"]}: solar_scale must be a positive number')
        if solar_scale!=1.0 and base.get("atmosphere"):
            # The NN path radiance is baked for the unscaled sun: the same
            # factor applied to the solar LUT must damp its solar-scatter
            # term, or a scaled-sun scene gets an unscaled twilight fog.
            base["atmosphere"]["solar_scatter_scale"]=solar_scale
        if (d['solar_ir_policy']=='planck_tail' or solar_scale!=1.0) and base.get('lighting',{}).get('solar_lut'):
            solar=out/'provenance'/('solar_'+definition['id']+'.csv')
            # The source hash is part of the frozen recipe; this derived file is
            # deterministic and also frozen via each scene TOML and its hash.
            derived[solar.relative_to(out).as_posix()]=solar_extension(native_path(base['lighting']['solar_lut']),solar,solar_scale)
        scene_info.append({'id':definition['id'],'source_config':canonical_path(source),'model':canonical_path(model),'target_world_units':target,'radius_m':radius,
            'split_group':model_hash,'overrides':definition.get('overrides',{}),'assumptions':{'thermal':d['thermal_mode'],'atmosphere':d['atmosphere_mode'],
            'ir_materials':('merged spectral library bound across VIS..LWIR; emissivity follows Kirchhoff from measured reflectance; unbound materials keep authored ir_emissivity (default 0.9)'
                            if d['ir_materials']=='spectral' else
                            'constant_scene_authored_emissivity; unmatched materials default 0.9; original geometry/textures retained'),
            'ir_refractive_index':('scene nk bindings kept only where source data covers the band; bound materials otherwise use measured reflectance' if d['ir_materials']=='spectral'
                                  else 'scene/default real index; source nk bindings omitted in IR'),
            'ir_solar_tail':'5778 K Planck shape anchored to final source CSV row' if solar else 'source CSV unchanged',
            'solar_scale':solar_scale,
            'vis_response':'authored generic RGB QE curves, Bayer RGGB; not measured camera calibration'}})
        for ai,ei,ri,rep in product(range(len(settings['sampling']['azimuths_deg'])),range(len(settings['sampling']['elevations_deg'])),range(len(settings['sampling']['radius_scales'])),range(settings['sampling']['repeats'])):
            az=settings['sampling']['azimuths_deg'][ai];el=settings['sampling']['elevations_deg'][ei];rs=settings['sampling']['radius_scales'][ri]
            a,e=math.radians(az),math.radians(el);r=radius*rs/ws
            position=add(target,[r*math.cos(e)*math.sin(a),r*math.sin(e),r*math.cos(e)*math.cos(a)])
            rig=pose(position,target,[0,1,0])
            identity=f'{definition["id"]}_a{ai:03}_e{ei:02}_r{ri:02}_n{rep:02}'
            sample={'sample_id':identity,'scene_id':definition['id'],'split_group':model_hash,'parameters':{'azimuth_deg':az,'elevation_deg':el,'radius_scale':rs,'repeat':rep},'rig_to_world':rig,'cameras':{}}
            for band in BANDS:
                c=settings['cameras'][band];p=camera_pose(rig,c,ws)
                if band=='vis' and 'vis_exposure_s' in definition:
                    # Low-light arms take a realistic long exposure so VIS is
                    # SNR-limited rather than photon-starved at the day value.
                    c=dict(c);c['exposure_s']=definition['vis_exposure_s']
                seed=int(fingerprint([d['seed'],identity,band])[:8],16)
                # per_acquisition: the detector seed stays one-per-band (FPN
                # maps hold still, like a real camera) while this counter
                # shifts the temporal noise classes' stream.
                acq=int(fingerprint([d['seed'],'acquisition',identity,band])[:8],16) if d['noise_seed_policy']=='per_acquisition' else None
                doc=scene_for_band(base,band,materials,d,solar,acq)
                doc['renderer'].update(resolution=c['resolution'],seed=seed,output='frame.exr')
                doc['camera']={'position':[p[3],p[7],p[11]],'look_at':[p[3]+p[2],p[7]+p[6],p[11]+p[10]],'up':[-p[1],-p[5],-p[9]],'projection':'perspective','fov_y':c['fov_y_deg']}
                doc['sensor']=sensor_config(band,c,d,identity)
                doc['isp']={'auto_exposure':False,'auto_white_balance':False}
                relative=f'jobs/{identity}/{band}.toml';text=toml_text(doc);texts[relative]=text
                sample['cameras'][band]={'job':relative,'job_sha256':hashlib.sha256(text.encode()).hexdigest(),'seed':seed,'acquisition_index':acq,'camera_to_world':p,'profile':c,
                    'sensor_expected':doc['sensor'],'spectral_mode':MODES[band]}
            samples.append(sample)
    def hashed(paths):
        # Sort by canonical path string: Path ordering is case-sensitive on
        # POSIX and case-folding on Windows, which would fork the fingerprint.
        filemap={}
        for path in paths:
            for p in (path.rglob('*') if path.is_dir() else iter((path,))):
                if p.is_file():filemap[canonical_path(p)]=p
        return [{'path':k,'sha256':digest(filemap[k]),'size_bytes':filemap[k].stat().st_size} for k in sorted(filemap)]
    closure=hashed(assets);observed=hashed(renderer_files)
    # Physics-relevant renderer sources gate with the version string. The
    # version contract alone cannot catch a same-version rebuild whose
    # physics changed -- exactly the failure mode that once mixed pre- and
    # post-fix LWIR under 0.7.3 -- so shader, atmosphere-baking and
    # sensor-chain sources are fingerprinted too. For an installed CLI
    # without a sibling source tree the list is empty and the version string
    # carries the contract alone, as documented.
    physics_roots=[cli_root/'src'/'shaders',cli_root/'src'/'libQuantiloom'/'atmos',
                   cli_root/'src'/'libQuantiloom'/'postprocess']
    physics_sources=hashed({p for p in physics_roots if p.is_dir()})
    if POSIX:
        unreachable=[e['path'] for e in closure if not is_drive_path(e['path'])]
        if unreachable:
            print(f'WARNING: assets not reachable by the Windows CLI: {unreachable}',flush=True)
    version=subprocess.check_output([str(cli),'--version'],text=True).strip()
    recipe={'settings':settings,'sources':closure,'cli_version':version,'generator_sha256':digest(Path(__file__)),
            'physics_sources':physics_sources,
            'derived_assets':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in derived.items()}}
    identity=fingerprint(recipe)
    result={'schema':SCHEMA,'schema_version':VERSION,'dataset_id':identity,'name':d['name'],'recipe':recipe,'renderer_observed':observed,
            'scenes':scene_info,'samples':samples,
            'registration':'unregistered_native_camera_grids','geometry_ground_truth':d['geometry_truth'],
            'fixed_camera_profiles':settings['cameras'],'parameters_vary':'scene, rig azimuth/elevation/radius, repeat; each parameter tuple includes all three bands'}
    destination=out/'dataset.plan.json'
    if destination.exists() and load_json(destination)['dataset_id']!=identity:
        raise ValueError('existing dataset has a different recipe, source asset, CLI version or generator fingerprint; choose a new output directory')
    for relative,text in {**derived,**texts}.items():
        path=out/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=text.encode():raise ValueError(f'planned file changed: {path}')
        path.write_text(text,encoding='utf-8',newline='\n')
    atomic_json(destination,result)
    print(f'Planned {len(samples)} triplets / {len(samples)*3} camera captures in {out}',flush=True)
    return result


def safe_path(root,relative):
    if not isinstance(relative,str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('invalid relative dataset path')
    p=Path(relative)
    if p.is_absolute() or any(part in ('.','..') for part in relative.split('/')):raise ValueError('unsafe dataset path')
    path=root/p
    if not path.resolve().is_relative_to(root.resolve()):raise ValueError('dataset path escapes package')
    return path


def check_expected(actual,expected,path='sensor'):
    # Presets and serialization add fields; only explicit requested parameters
    # are compared. A misspelled requested parameter cannot silently disappear.
    for k,v in expected.items():
        if k=='preset':continue
        if k not in actual:raise ValueError(f'effective configuration is missing {path}.{k}')
        a=actual[k]
        if isinstance(v,dict):check_expected(a,v,path+'.'+k)
        elif isinstance(v,list) and v and isinstance(v[0],dict):
            if len(a)!=len(v):raise ValueError('effective channel count mismatch')
            for i,(av,ev) in enumerate(zip(a,v)):check_expected(av,ev,f'{path}.{k}[{i}]')
        elif isinstance(v,(float,int)) and not isinstance(v,bool):
            if not math.isclose(a,v,rel_tol=1e-8,abs_tol=1e-12):raise ValueError(f'effective {path}.{k}: requested {v}, actual {a}')
        elif a!=v:raise ValueError(f'effective {path}.{k} mismatch: {a!r} vs {v!r}')


def verify_capture(record_path,camera,inspect_pixels=False):
    record=load_json(record_path)
    if record.get('state')!='complete' or record.get('capture_status')!='complete':raise ValueError('incomplete render transaction')
    if record.get('schema')!='quantiloom.dataset.export' or record.get('schema_version') not in (1,2):raise ValueError('unsupported export schema')
    if not re.fullmatch('[0-9a-f]{64}',record.get('record_id','')):raise ValueError('invalid record identity')
    root=record_path.parent;seen=set();ids=set();selected={};qa={}
    for entry in [record['replay'],*record['products']]:
        p=safe_path(root,entry['path'])
        if entry['path'].casefold() in seen:raise ValueError('duplicate product path')
        seen.add(entry['path'].casefold())
        if not p.is_file() or digest(p)!=entry['sha256']:raise ValueError(f'missing/corrupt product: {p}')
        if 'size_bytes' in entry and p.stat().st_size!=entry['size_bytes']:raise ValueError('product size mismatch')
        if 'product_id' in entry:
            if entry['product_id'] in ids:raise ValueError('duplicate product ID')
            ids.add(entry['product_id'])
    for product in record['products']:
        name=product['path'];desc=product['description']
        if name in ('frame_measurement.exr','frame_rawdn.exr','frame_corrected.exr','frame_display.exr'):
            key=name.removeprefix('frame_').removesuffix('.exr')
            if [desc['width'],desc['height']]!=camera['profile']['resolution']:raise ValueError('native dimensions changed')
            signal=desc['signal']
            if not signal.get('unit') or not signal.get('kind'):raise ValueError('missing signal semantics')
            actual=tomllib.loads(desc['provenance']['camera_config_toml'])['sensor']
            check_expected(actual,camera['sensor_expected'])
            geometry=desc['provenance']['geometry']
            if geometry['camera_model']!='brown_conrady' or geometry['camera_axes']!='right_down_forward':raise ValueError('unexpected native geometry convention')
            if len(geometry['camera_to_world'])!=16 or len(geometry['intrinsics'])!=9:raise ValueError('incomplete native geometry matrices')
            if any(not math.isclose(a,b,rel_tol=2e-6,abs_tol=2e-6) for a,b in zip(geometry['camera_to_world'],camera['camera_to_world'])):raise ValueError('effective camera pose mismatch')
            w,h=camera['profile']['resolution'];f=h/(2*math.tan(math.radians(camera['profile']['fov_y_deg'])/2))
            expected_k=[f,0,w/2,0,f,h/2,0,0,1]
            if any(not math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-8) for a,b in zip(geometry['intrinsics'],expected_k)):raise ValueError('effective FOV/intrinsics mismatch')
            selected[key]={'path':name,'signal':signal,'provenance':desc['provenance'],'image_metadata':desc.get('image_metadata',{})}
            if inspect_pixels:
                import OpenEXR
                import numpy as np
                with OpenEXR.File(str(root/name),separate_channels=True) as f:
                    if set(f.channels())!=set(desc['channel_names']):raise ValueError('EXR channel mismatch')
                    arrays=[f.channels()[n].pixels for n in desc['channel_names']]
                    a=np.stack(arrays,axis=-1)
                    if list(a.shape[:2])!=[desc['height'],desc['width']] or not np.isfinite(a).all():raise ValueError('invalid EXR pixels')
                    for k in ('quantiloom_record_id','quantiloom_product_id'):
                        expected=record['record_id'] if k.endswith('record_id') else product['product_id']
                        if f.header().get(k)!=expected:raise ValueError('EXR transaction identity mismatch')
                qa[key]={'min':float(a.min()),'max':float(a.max()),'std':float(a.std()),'flat':bool(a.std()==0)}
                if key=='rawdn':
                    adc=(1<<camera['profile']['adc_bits'])-1
                    qa[key]['adc_max_fraction']=float(np.mean(a>=adc))
                    if 'full_well_e' in camera['profile']:
                        ceiling=min(adc,math.floor(camera['profile']['full_well_e']/camera['profile']['electrons_per_dn']))
                        qa[key]['full_well_or_adc_fraction']=float(np.mean(a>=ceiling))
    if set(selected)!={'measurement','rawdn','corrected','display'}:raise ValueError('missing required sensor product(s)')
    reference=next((p for p in record['products'] if p['path']=='frame.exr'),None)
    if not reference:raise ValueError('missing main render reference')
    selected['rendered_reference']={'path':'frame.exr','role':'separate_ideal_pinhole_render_without_sensor_degradation',
        'description':reference['description']}
    # Products beyond the required four (truth maps, apparent temperature,
    # previews) are hash-verified and copied like everything else; their
    # descriptions stay in the record and their names are surfaced here.
    known={'frame.exr',*(f'frame_{k}.exr' for k in ('measurement','rawdn','corrected','display'))}
    extras={p['path']:p['description'] for p in record['products'] if p['path'] not in known}
    return {'record_id':record['record_id'],'products':selected,'pixel_qa':qa,'extra_products':extras}


def receipt_path(out,sample,band):return out/'state'/sample['sample_id']/(band+'.json')


def valid_receipt(out,sample,band,pixels):
    path=receipt_path(out,sample,band)
    if not path.exists():return None
    receipt=load_json(path)
    if receipt.get('status')!='complete':return None
    camera=sample['cameras'][band]
    if receipt['job_sha256']!=camera['job_sha256']:raise ValueError('receipt recipe mismatch')
    record=safe_path(out,receipt['record'])
    if digest(record)!=receipt['record_sha256']:raise ValueError('sidecar hash mismatch')
    summary=verify_capture(record,camera,pixels)
    return receipt,summary


def publish(out,sample,band,attempt,camera,pixels):
    record_path=attempt/'frame.metadata.json'
    summary=verify_capture(record_path,camera,pixels)
    destination=out/'samples'/sample['sample_id']/band/('v'+attempt.name.rsplit('-',1)[-1])
    if destination.exists():
        verify_capture(destination/record_path.name,camera,pixels)
        if digest(destination/record_path.name)!=digest(record_path):raise ValueError('publication version does not match attempt')
    else:
        temporary=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.tmp')
        temporary.mkdir(parents=True)
        record=load_json(record_path)
        for artifact in [record['replay'],*record['products']]:
            target=safe_path(temporary,artifact['path']);target.parent.mkdir(parents=True,exist_ok=True)
            # Copy file content, not the renderer's Windows owner-only ACLs.
            shutil.copyfile(safe_path(attempt,artifact['path']),target)
        shutil.copyfile(record_path,temporary/record_path.name)
        verify_capture(temporary/record_path.name,camera,pixels)
        os.replace(temporary,destination)
    receipt={'status':'complete','job_sha256':camera['job_sha256'],'record':(destination/record_path.name).relative_to(out).as_posix(),
             'record_sha256':digest(destination/record_path.name),'record_id':summary['record_id'],'verified':'pixels_and_hashes' if pixels else 'hashes_and_effective_parameters','pixel_qa':summary['pixel_qa']}
    atomic_json(receipt_path(out,sample,band),receipt)
    return receipt,summary


def sample_manifest(out,plan,sample,results):
    if set(results)!=set(BANDS):return None
    manifest={'schema':'quantiloomfusion.sample','schema_version':1,'dataset_id':plan['dataset_id'],'sample_id':sample['sample_id'],
              'scene_id':sample['scene_id'],'split_group':sample['split_group'],'state':'complete','parameters':sample['parameters'],
              'rig_to_world':sample['rig_to_world'],'registration':'unregistered',
              'geometry_ground_truth':plan['recipe']['settings']['dataset']['geometry_truth'],'observations':{}}
    for band,(receipt,summary) in results.items():
        base=Path(receipt['record']).parent
        manifest['observations'][band]={'record':receipt['record'],'record_sha256':receipt['record_sha256'],'camera_to_world':sample['cameras'][band]['camera_to_world'],
            'spectral_mode':MODES[band],'products':{k:{**v,'path':(base/v['path']).as_posix()} for k,v in summary['products'].items()},
            'extra_products':{(base/k).as_posix():v for k,v in summary['extra_products'].items()},'pixel_qa':summary['pixel_qa']}
    path=out/'samples'/sample['sample_id']/'sample.json';atomic_json(path,manifest)
    return path.relative_to(out).as_posix()


def update_index(out,plan):
    complete=[]
    for s in plan['samples']:
        p=out/'samples'/s['sample_id']/'sample.json'
        if p.exists():
            m=load_json(p)
            if m['dataset_id']==plan['dataset_id'] and m['state']=='complete':complete.append({'sample_id':s['sample_id'],'manifest':p.relative_to(out).as_posix(),'split_group':s['split_group']})
    atomic_json(out/'dataset.index.json',{'schema':SCHEMA,'schema_version':VERSION,'dataset_id':plan['dataset_id'],'complete_triplets':len(complete),'planned_triplets':len(plan['samples']),'samples':complete})


def run_dataset(out,plan,args,lease,lock):
    pixels=plan['recipe']['settings']['dataset']['verification']=='pixels'
    if pixels:
        try:import OpenEXR,numpy
        except ImportError as e:raise ValueError('pixel verification requires OpenEXR and numpy; install them or choose verification="hashes"') from e
    cli=native_path(plan['recipe']['settings']['dataset']['cli']).resolve()
    if POSIX and not is_drive_path(canonical_path(out)):
        raise ValueError('output directory must be reachable by the Windows CLI; '
                         'on WSL use a path under /mnt/<drive>/')
    captures=0;skipped=0;failed=0;completed=0
    audit_every=plan['recipe']['settings']['dataset'].get('closure_audit_every',0)
    selected=[s for s in plan['samples'] if not args.scene or s['scene_id'] in args.scene]
    if args.scene and not selected:raise ValueError('no selected scenes in plan')
    try:
        try:
            for sample in selected:
                results={};changed=False;lwir_attempt=None
                for band in BANDS:
                    try:existing=valid_receipt(out,sample,band,pixels)
                    except (ValueError,OSError,KeyError) as e:
                        print(f'REPAIR {sample["sample_id"]}/{band}: {e}',flush=True);existing=None
                        stale=out/'samples'/sample['sample_id']/'sample.json'
                        if stale.exists():stale.unlink()
                        update_index(out,plan)
                    if existing:
                        results[band]=existing;skipped+=1;continue
                    if args.max_captures is not None and captures>=args.max_captures:
                        print(f'Stopped at capture budget; complete triplets only are indexed. Rendered={captures}, skipped={skipped}',flush=True)
                        return 1 if failed else 0
                    camera=sample['cameras'][band];job=safe_path(out,camera['job'])
                    if digest(job)!=camera['job_sha256']:raise ValueError('planned job was edited')
                    attempts=out/'attempts'/sample['sample_id']/band;attempts.mkdir(parents=True,exist_ok=True)
                    number=max([int(p.name.split('-')[-1]) for p in attempts.glob('attempt-*') if p.is_dir()]+[0])+1
                    # Recover a fully rendered but not yet published acquisition.
                    previous=attempts/f'attempt-{number-1:04}'
                    if number>1 and (previous/'frame.metadata.json').exists():
                        try:
                            results[band]=publish(out,sample,band,previous,camera,pixels)
                            print(f'RECOVER {sample["sample_id"]}/{band}',flush=True)
                            if band=='lwir':lwir_attempt=previous
                            continue
                        except (ValueError,OSError,KeyError):pass
                    attempt=attempts/f'attempt-{number:04}';attempt.mkdir()
                    doc=tomllib.loads(job.read_text(encoding='utf-8'));doc['renderer']['output']=canonical_path(attempt/'frame.exr')
                    actual=attempt/'capture.toml';actual.write_text(toml_text(doc),encoding='utf-8')
                    state=receipt_path(out,sample,band)
                    atomic_json(state,{'status':'running','job_sha256':camera['job_sha256'],'attempt':attempt.relative_to(out).as_posix()})
                    print(f'RENDER {sample["sample_id"]}/{band}',flush=True)
                    with (attempt/'render.log').open('w',encoding='utf-8') as log:
                        child=subprocess.Popen([str(cli),canonical_path(actual)],cwd=attempt,stdout=log,stderr=subprocess.STDOUT)
                        lease['child_pid']=child.pid;atomic_json(lock,lease)
                        try:code=child.wait()
                        except KeyboardInterrupt:
                            child.terminate()
                            try:child.wait(timeout=10)
                            except subprocess.TimeoutExpired:child.kill();child.wait()
                            atomic_json(state,{'status':'interrupted','job_sha256':camera['job_sha256'],'attempt':attempt.relative_to(out).as_posix()})
                            raise
                        finally:lease['child_pid']=None;atomic_json(lock,lease)
                    captures+=1
                    try:
                        if code:raise ValueError(f'CLI exit {code}; see {attempt/"render.log"}')
                        results[band]=publish(out,sample,band,attempt,camera,pixels);changed=True
                        if band=='lwir':lwir_attempt=attempt
                        qa=results[band][1]['pixel_qa']
                        raw=qa.get('rawdn',{})
                        if max(raw.get('adc_max_fraction',0),raw.get('full_well_or_adc_fraction',0))>.05:
                            print(f'QA warning: {sample["sample_id"]}/{band} has >5% RAW at full-well or ADC limit',flush=True)
                    except (ValueError,OSError,KeyError) as e:
                        failed+=1;atomic_json(state,{'status':'failed','error':str(e),'job_sha256':camera['job_sha256'],'attempt':attempt.relative_to(out).as_posix()})
                        print(f'FAILED {sample["sample_id"]}/{band}: {e}',flush=True)
                        if not args.keep_going:return 1
                if sample_manifest(out,plan,sample,results):
                    update_index(out,plan)
                    print(f'COMPLETE {sample["sample_id"]}',flush=True)
                    completed+=1
                    if audit_every and completed%audit_every==0 and lwir_attempt is not None:
                        # Periodic radiometric closure: recomputing the NN
                        # atmosphere offline and checking L_pred vs corrected
                        # catches mid-run renderer regressions long before
                        # verify does.
                        audit=Path(__file__).resolve().parent/'_closure_audit.py'
                        try:
                            proc=subprocess.run([sys.executable,str(audit),str(lwir_attempt)],
                                capture_output=True,text=True,timeout=300)
                            report=(proc.stdout+proc.stderr).strip()
                            (lwir_attempt/'closure_audit.txt').write_text(report+"\n",encoding='utf-8')
                            summary=[l for l in report.splitlines() if 'closure residual' in l]
                            print(f'CLOSURE {sample["sample_id"]}: {summary[0] if summary else report.splitlines()[-1] if report else "no output"}',flush=True)
                        except Exception as e:
                            print(f'CLOSURE {sample["sample_id"]}: audit unavailable ({e})',flush=True)
            print(f'Finished: rendered={captures}, skipped={skipped}, failed={failed}',flush=True)
            return 1 if failed else 0
        finally:update_index(out,plan)
    finally:
        lease['child_pid']=None


def verify_dataset(out,pixels=False):
    plan=load_json(out/'dataset.plan.json');complete=partial=0;completed_ids=[]
    for s in plan['samples']:
        results={}
        for band in BANDS:
            found=valid_receipt(out,s,band,pixels)
            if found:results[band]=found
        if len(results)==3:
            manifest=load_json(out/'samples'/s['sample_id']/'sample.json')
            if manifest['dataset_id']!=plan['dataset_id'] or manifest['state']!='complete':raise ValueError('sample identity mismatch')
            for band,(receipt,summary) in results.items():
                observation=manifest['observations'][band]
                if observation['record_sha256']!=receipt['record_sha256'] or observation['record']!=receipt['record']:raise ValueError('sample/receipt mismatch')
                for k,v in summary['products'].items():
                    if observation['products'][k]['path']!=(Path(receipt['record']).parent/v['path']).as_posix():raise ValueError('sample product path mismatch')
            complete+=1
            completed_ids.append(s['sample_id'])
        elif results:partial+=1
    index=load_json(out/'dataset.index.json') if (out/'dataset.index.json').exists() else {'complete_triplets':0}
    if index['complete_triplets']!=complete:raise ValueError('index completion count mismatch')
    if [s['sample_id'] for s in index.get('samples',[])]!=completed_ids:raise ValueError('index sample membership mismatch')
    print(f'Verified: complete={complete}, partial={partial}, planned={len(plan["samples"])}',flush=True)
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('plan','run'):
        p=sub.add_parser(name);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
        if name=='run':
            p.add_argument('--scene',action='append',help='select a scene; repeatable')
            p.add_argument('--max-captures',type=int,help='new camera captures this run; may stop between bands')
            p.add_argument('--keep-going',action='store_true')
    p=sub.add_parser('verify');p.add_argument('--out',type=Path,required=True);p.add_argument('--inspect-exr',action='store_true')
    args=parser.parse_args(argv);out=args.out.resolve()
    try:
        if args.command=='verify':return verify_dataset(out,args.inspect_exr)
        if args.command=='run' and args.max_captures is not None and args.max_captures<1:raise ValueError('max-captures must be positive')
        out.mkdir(parents=True,exist_ok=True)
        with runner_lock(out) as (lease,lock):
            planned=plan(args.config.resolve(),out)
            if args.command=='run':return run_dataset(out,planned,args,lease,lock)
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Run the same command to resume.',file=sys.stderr);return 130
    except (ValueError,OSError,KeyError,subprocess.CalledProcessError) as e:
        print(f'ERROR: {e}',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
