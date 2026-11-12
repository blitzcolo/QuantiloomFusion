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
from .constants import BANDS, MODES, REFERENCE_PREFIX, SCHEMA, VERSION
from .fingerprint import generator_files, generator_fingerprint
from .renderer import QuantiloomCli
from .scene import canonicalize, read_gltf, reject_motion, resolve_asset, scene_for_band
from .sensor import sensor_config
from .settings import positive, read_settings
from .solar import forcing_direct, forcing_row, reference_direct, solar_extension

# [atmosphere] keys the air derivation owns. rh is the fraction the renderer
# resolves both humidity keys to; t_ground_K is the air temperature its NN
# atmosphere and thermal sky use; air_temperature_k feeds the clear-sky model.
AIR_KEYS=('t_ground_K','air_temperature_k','rh','relative_humidity')


def _agrees(a,b,path=''):
    # First differing key path, or None. Numbers compare with a tolerance that
    # covers the last-ULP trig spread between Windows and Linux libm; keys
    # named job_sha256 are skipped because they hash the job text, which
    # legitimately differs at that scale.
    if isinstance(a,dict) and isinstance(b,dict):
        ka={k for k in a if k!='job_sha256'}
        if ka!={k for k in b if k!='job_sha256'}:return path or '<root>'
        for k in sorted(ka):
            diff=_agrees(a[k],b[k],f'{path}.{k}' if path else k)
            if diff:return diff
        return None
    if isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):return path
        for i,(x,y) in enumerate(zip(a,b)):
            diff=_agrees(x,y,f'{path}[{i}]')
            if diff:return diff
        return None
    if isinstance(a,(int,float)) and isinstance(b,(int,float)) and not isinstance(a,bool) and not isinstance(b,bool):
        return None if math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) else path
    return None if a==b else path


def plan(config: Path,out: Path,renderer=None):
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
    subset=settings.get('reference_subset')
    out.mkdir(parents=True,exist_ok=True)
    if d.get("atmosphere_model_pack"):
        pack=native_path(d["atmosphere_model_pack"]).resolve()
        if not pack.is_dir():raise FileNotFoundError(f"atmosphere_model_pack not found: {pack}")
        assets.add(pack)
    for definition in settings["scenes"]:
        source=resolve_asset(definition["config"],qt/'assets/configs',qt,cli_root)
        assets.add(source)
        base=tomllib.loads(source.read_text(encoding="utf-8-sig"))
        # The hour a scene was composed for lives in its own [thermal], which a
        # night or low-light override replaces; keep it for the solar scale.
        source_thermal=dict(base.get('thermal',{}))
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
        arm_scale=solar_scale
        if arm_scale!=1.0 and base.get("atmosphere"):
            # The NN path radiance is baked for the unscaled sun: the same
            # factor applied to the solar LUT must damp its solar-scatter
            # term, or a scaled-sun scene gets an unscaled twilight fog. The
            # forcing match below only aligns the LUT's amplitude with the
            # beam that heated the ground, so it does not touch the NN.
            base["atmosphere"]["solar_scatter_scale"]=arm_scale
        derivation={'policy':d['daylight_solar_scale']}
        lighting=base.get('lighting',{})
        if d['daylight_solar_scale']=='forcing_direct' and lighting.get('solar_lut'):
            # The reference LUT has one fixed beam (G-173: ~900 W/m2 at AM1.5)
            # while the thermal solve heats the ground with the forcing file's
            # beam at the scene's hour; scale the LUT to that beam so the sun
            # that lights the frame is the sun that warmed it. A ratio, so the
            # renderer's derived short-wave absorptivity is unchanged.
            if not source_thermal.get('forcing_file') or 'time_h' not in source_thermal:
                raise ValueError(f'scene {definition["id"]}: daylight_solar_scale="forcing_direct" needs [thermal] forcing_file and time_h in {source.name}')
            columns=lighting.get('solar_lut_columns')
            if not isinstance(columns,list) or not columns:
                raise ValueError(f'scene {definition["id"]}: daylight_solar_scale="forcing_direct" needs lighting.solar_lut_columns')
            forcing=resolve_asset(source_thermal['forcing_file'],source.parent,qt,cli_root);assets.add(forcing)
            lut=native_path(lighting['solar_lut'])
            measured=forcing_direct(forcing,float(source_thermal['time_h']))
            reference=reference_direct(lut,columns[0])
            derivation.update(forcing_file=canonical_path(forcing),time_h=float(source_thermal['time_h']),forcing_direct_w_m2=measured,
                reference_lut=canonical_path(lut),reference_column=columns[0],reference_direct_w_m2=reference,
                forcing_ratio=measured/reference,arm_scale=arm_scale)
            solar_scale=arm_scale*measured/reference
        air={'policy':d['atmosphere_air']}
        if d['atmosphere_air']=='forcing_row' and d['atmosphere_mode']=='source' and base.get('atmosphere'):
            # The thermal solve heats the ground with the forcing file's air and
            # humidity at the entry's hour, while the renderer's sky and path
            # radiance follow [atmosphere]; derive the latter from the same row
            # for every entry, day and night alike, so the air that warmed the
            # ground is the air the camera looks through. Presets (aerosol,
            # visibility) stay as the scene states them.
            thermal=base.get('thermal',{})
            if not thermal.get('forcing_file') or 'time_h' not in thermal:
                raise ValueError(f'scene {definition["id"]}: atmosphere_air="forcing_row" needs [thermal] forcing_file and time_h')
            stated=[k for k in AIR_KEYS if k in definition.get('overrides',{}).get('atmosphere',{})]
            if stated:
                raise ValueError(f'scene {definition["id"]}: [atmosphere] states {stated} by hand; atmosphere_air="forcing_row" derives them from the forcing file')
            forcing=resolve_asset(thermal['forcing_file'],source.parent,qt,cli_root);assets.add(forcing)
            row=forcing_row(forcing,float(thermal['time_h']))
            if 'relative_humidity' not in row:raise ValueError(f'{forcing}: atmosphere_air="forcing_row" needs the relative_humidity column')
            written={'t_ground_K':row['air_temperature_k'],'air_temperature_k':row['air_temperature_k'],'rh':round(row['relative_humidity']/100,12)}
            # A copy: the override table is part of the recorded settings.
            base['atmosphere']={**{k:v for k,v in base['atmosphere'].items() if k not in AIR_KEYS},**written}
            air.update(forcing_file=canonical_path(forcing),time_h=float(thermal['time_h']),source_lines=row['source_lines'],
                air_temperature_k=row['air_temperature_k'],relative_humidity_pct=row['relative_humidity'],written=written)
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
            'solar_scale':solar_scale,'solar_scale_derivation':derivation,'atmosphere_air':air,
            'vis_response':'authored generic RGB QE curves, Bayer RGGB; not measured camera calibration'}})
        for ai,ei,ri,rep in product(range(len(settings['sampling']['azimuths_deg'])),range(len(settings['sampling']['elevations_deg'])),range(len(settings['sampling']['radius_scales'])),range(settings['sampling']['repeats'])):
            az=settings['sampling']['azimuths_deg'][ai];el=settings['sampling']['elevations_deg'][ei];rs=settings['sampling']['radius_scales'][ri]
            a,e=math.radians(az),math.radians(el);r=radius*rs/ws
            position=add(target,[r*math.cos(e)*math.sin(a),r*math.sin(e),r*math.cos(e)*math.cos(a)])
            rig=pose(position,target,[0,1,0])
            identity=f'{definition["id"]}_a{ai:03}_e{ei:02}_r{ri:02}_n{rep:02}'
            sample={'sample_id':identity,'scene_id':definition['id'],'split_group':model_hash,'parameters':{'azimuth_deg':az,'elevation_deg':el,'radius_scale':rs,'repeat':rep},'rig_to_world':rig,'cameras':{}}
            def capture(band,key,c,p):
                # key names the capture: the band itself, or ref_<band> for a
                # co-view reference; seeds and the job path follow the key.
                seed=int(fingerprint([d['seed'],identity,key])[:8],16)
                # per_acquisition: the detector seed stays one-per-band (FPN
                # maps hold still, like a real camera) while this counter
                # shifts the temporal noise classes' stream.
                acq=int(fingerprint([d['seed'],'acquisition',identity,key])[:8],16) if d['noise_seed_policy']=='per_acquisition' else None
                doc=scene_for_band(base,band,materials,d,solar,acq)
                doc['renderer'].update(resolution=c['resolution'],seed=seed,output='frame.exr')
                # The renderer rebuilds the view basis in float32 from
                # look_at - position; a point one unit ahead loses the basis to
                # rounding hundreds of units from the origin, so aim at range r.
                doc['camera']={'position':[p[3],p[7],p[11]],'look_at':[p[3]+r*p[2],p[7]+r*p[6],p[11]+r*p[10]],'up':[-p[1],-p[5],-p[9]],'projection':'perspective','fov_y':c['fov_y_deg']}
                doc['sensor']=sensor_config(band,c,d,identity)
                doc['isp']={'auto_exposure':False,'auto_white_balance':False}
                relative=f'jobs/{identity}/{key}.toml';text=toml_text(doc);texts[relative]=text
                return {'job':relative,'job_sha256':hashlib.sha256(text.encode()).hexdigest(),'seed':seed,'acquisition_index':acq,'camera_to_world':p,'profile':c,
                    'sensor_expected':doc['sensor'],'spectral_mode':MODES[band]}
            for band in BANDS:
                c=settings['cameras'][band];p=camera_pose(rig,c,ws)
                if band=='vis' and 'vis_exposure_s' in definition:
                    # Low-light arms take a realistic long exposure so VIS is
                    # SNR-limited rather than photon-starved at the day value.
                    c=dict(c);c['exposure_s']=definition['vis_exposure_s']
                sample['cameras'][band]=capture(band,band,c,p)
            if subset and az in subset['azimuths_deg'] and el in subset['elevations_deg'] and rs in subset['radius_scales']:
                # Co-view references: every band rendered noise-free through
                # the VIS camera's pose, resolution, FOV, distortion and PSF,
                # so all three are pixel-aligned with VIS (no parallax). The
                # detector, spectral response and radiometry stay the band's
                # own: the focal length in mm follows the band's pixel pitch
                # while the intrinsics in pixels equal VIS's.
                vis=sample['cameras']['vis']
                geometry={k:vis['profile'][k] for k in ('resolution','fov_y_deg','distortion','offset_rdf_m','rotation_ypr_deg','psf_sigma_px')}
                sample['references']={}
                for band in BANDS:
                    c={**sample['cameras'][band]['profile'],**geometry,'noise_free':True}
                    record=capture(band,REFERENCE_PREFIX+band,c,vis['camera_to_world'])
                    sample['references'][band]={**record,'geometry_from':'vis','psf_sigma_px':geometry['psf_sigma_px']}
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
    version=(renderer or QuantiloomCli(cli)).version()
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
    references=sum(len(s.get('references',{})) for s in samples)
    extra=f' (+{references} reference renders)' if references else ''
    stored=load_json(destination) if destination.exists() else None
    if stored is not None and stored['dataset_id']!=identity:
        raise ValueError('existing dataset has a different recipe, source asset, CLI version or generator fingerprint; choose a new output directory')
    if stored is not None:
        # Trig results differ in the last ULP between Windows and Linux libm,
        # so the first planning host freezes the plan: a tolerant re-plan keeps
        # the stored job files and plan rather than churning their hashes.
        # renderer_observed is recorded but not gated: compiled blobs churn.
        diff=_agrees({k:v for k,v in stored.items() if k!='renderer_observed'},
                     {k:v for k,v in result.items() if k!='renderer_observed'})
        if diff:raise ValueError(f'planned file changed: {destination}: {diff}')
        expected={c['job']:c['job_sha256'] for s in stored['samples'] for c in [*s['cameras'].values(),*s.get('references',{}).values()]}
        for relative,text in {**derived,**texts}.items():
            path=out/relative;path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():path.write_text(text,encoding='utf-8',newline='\n')
            elif path.read_bytes()!=text.encode():
                if relative in derived:raise ValueError(f'planned file changed: {path}')
                diff=_agrees(tomllib.loads(path.read_text(encoding='utf-8')),tomllib.loads(text))
                if diff:raise ValueError(f'planned file changed: {path}: {diff}')
            if relative in expected and digest(path)!=expected[relative]:raise ValueError(f'planned file changed: {path}')
        print(f'Planned {len(samples)} triplets / {len(samples)*3} camera captures{extra} in {out}',flush=True)
        return stored
    for relative,text in {**derived,**texts}.items():
        path=out/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=text.encode():raise ValueError(f'planned file changed: {path}')
        path.write_text(text,encoding='utf-8',newline='\n')
    atomic_json(destination,result)
    print(f'Planned {len(samples)} triplets / {len(samples)*3} camera captures{extra} in {out}',flush=True)
    return result
