"""Validation of a renderer export record (frame.metadata.json)."""

import math
import re
import tomllib
from pathlib import Path

from ..core.hashing import digest
from ..core.jsonio import load_json


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
