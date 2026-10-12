"""Dataset planning: recipe closure, sampling grid and job TOMLs."""

import hashlib
import math
import tomllib
from itertools import product
from pathlib import Path

from ..core.hashing import digest, fingerprint
from ..core.host import POSIX
from ..core.jsonio import atomic_json, load_json
from ..core.paths import canonical_path, is_drive_path, native_path
from ..core.tomlio import toml_text
from ..core.vec import add, camera_pose, norm, pose, scale
from .constants import BANDS, MODES, SCHEMA, VERSION
from .fingerprint import generator_files, generator_fingerprint
from .renderer import QuantiloomCli
from .scene import canonicalize, read_gltf, reject_motion, resolve_asset, scene_for_band
from .sensor import sensor_config
from .settings import positive, read_settings
from .solar import solar_extension


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
    version=QuantiloomCli(cli).version()
    files=generator_files()
    recipe={'settings':settings,'sources':closure,'cli_version':version,'generator_sha256':generator_fingerprint(files),'generator_files':files,
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
