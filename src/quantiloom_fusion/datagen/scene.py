"""Scene assembly: motion rejection, glTF reading, asset canonicalization."""

import copy
import json
import struct
from pathlib import Path

from ..core.jsonio import load_json
from ..core.paths import canonical_path, native_path
from .constants import MODES, PATH_KEYS


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
