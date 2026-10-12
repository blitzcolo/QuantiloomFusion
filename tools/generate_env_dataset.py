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
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tomllib
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantiloom_fusion.core.hashing import digest, fingerprint
from quantiloom_fusion.core.host import POSIX
from quantiloom_fusion.core.jsonio import atomic_json, load_json
from quantiloom_fusion.core.locking import alive, runner_lock
from quantiloom_fusion.core.paths import canonical_path, is_drive_path, native_path
from quantiloom_fusion.core.tomlio import scalar, toml_text
from quantiloom_fusion.core.vec import add, camera_pose, cross, dot, local_rotation, norm, pose, scale, unit
from quantiloom_fusion.datagen.constants import BANDS, MODES, PATH_KEYS, SCHEMA, VERSION
from quantiloom_fusion.datagen.export_record import check_expected, safe_path, verify_capture
from quantiloom_fusion.datagen.planner import plan
from quantiloom_fusion.datagen.renderer import CalledProcessError, QuantiloomCli, run_closure_audit
from quantiloom_fusion.datagen.scene import canonicalize, nk_span_nm, read_gltf, reject_motion, resolve_asset, scene_for_band
from quantiloom_fusion.datagen.settings import check_keys, positive, read_settings, vector
from quantiloom_fusion.datagen.solar import solar_extension
from quantiloom_fusion.datagen.sensor import sensor_config


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
    renderer=QuantiloomCli(cli)
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
                        child=renderer.start(actual,attempt,log)
                        lease['child_pid']=child.pid;atomic_json(lock,lease)
                        try:code=child.wait()
                        except KeyboardInterrupt:
                            child.terminate()
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
                            report=run_closure_audit(audit,lwir_attempt)
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
    except (ValueError,OSError,KeyError,CalledProcessError) as e:
        print(f'ERROR: {e}',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
