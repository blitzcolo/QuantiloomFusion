"""The run loop: render attempts, receipts, recovery, closure audits."""

import tomllib
from pathlib import Path

from ..core.hashing import digest
from ..core.host import POSIX
from ..core.jsonio import atomic_json
from ..core.paths import canonical_path, is_drive_path, native_path
from ..core.tomlio import toml_text
from .constants import BANDS, REFERENCE_PREFIX
from .export_record import safe_path
from .index import sample_manifest, update_index
from .publish import capture_record, publish, receipt_path, valid_receipt
from .renderer import QuantiloomCli, run_closure_audit


class _Halt(Exception):
    """Stop the run with an exit code (capture budget reached, or a failure without --keep-going)."""

    def __init__(self,code):
        super().__init__(code);self.code=code


def run_dataset(out,plan,args,lease,lock,renderer=None):
    pixels=plan['recipe']['settings']['dataset']['verification']=='pixels'
    if pixels:
        try:import OpenEXR,numpy
        except ImportError as e:raise ValueError('pixel verification requires OpenEXR and numpy; install them or choose verification="hashes"') from e
    cli=native_path(plan['recipe']['settings']['dataset']['cli']).resolve()
    renderer=renderer or QuantiloomCli(cli)
    if POSIX and not is_drive_path(canonical_path(out)):
        raise ValueError('output directory must be reachable by the Windows CLI; '
                         'on WSL use a path under /mnt/<drive>/')
    captures=0;skipped=0;failed=0;completed=0
    audit_every=plan['recipe']['settings']['dataset'].get('closure_audit_every',0)
    selected=[s for s in plan['samples'] if not args.scene or s['scene_id'] in args.scene]
    if args.scene and not selected:raise ValueError('no selected scenes in plan')

    def acquire(sample,key,on_repair):
        # One capture (a band, or ref_<band>): skip a verified receipt, recover
        # a rendered but unpublished attempt, else render a new attempt.
        # Returns ((receipt, summary), attempt) or (None, None) on failure.
        nonlocal captures,skipped,failed
        try:existing=valid_receipt(out,sample,key,pixels)
        except (ValueError,OSError,KeyError) as e:
            print(f'REPAIR {sample["sample_id"]}/{key}: {e}',flush=True);existing=None
            on_repair()
        if existing:
            skipped+=1;return existing,None
        if args.max_captures is not None and captures>=args.max_captures:
            print(f'Stopped at capture budget; complete triplets only are indexed. Rendered={captures}, skipped={skipped}',flush=True)
            raise _Halt(1 if failed else 0)
        camera=capture_record(sample,key);job=safe_path(out,camera['job'])
        if digest(job)!=camera['job_sha256']:raise ValueError('planned job was edited')
        attempts=out/'attempts'/sample['sample_id']/key;attempts.mkdir(parents=True,exist_ok=True)
        number=max([int(p.name.split('-')[-1]) for p in attempts.glob('attempt-*') if p.is_dir()]+[0])+1
        # Recover a fully rendered but not yet published acquisition.
        previous=attempts/f'attempt-{number-1:04}'
        if number>1 and (previous/'frame.metadata.json').exists():
            try:
                result=publish(out,sample,key,previous,camera,pixels)
                print(f'RECOVER {sample["sample_id"]}/{key}',flush=True)
                return result,previous
            except (ValueError,OSError,KeyError):pass
        attempt=attempts/f'attempt-{number:04}';attempt.mkdir()
        doc=tomllib.loads(job.read_text(encoding='utf-8'));doc['renderer']['output']=canonical_path(attempt/'frame.exr')
        actual=attempt/'capture.toml';actual.write_text(toml_text(doc),encoding='utf-8')
        state=receipt_path(out,sample,key)
        atomic_json(state,{'status':'running','job_sha256':camera['job_sha256'],'attempt':attempt.relative_to(out).as_posix()})
        print(f'RENDER {sample["sample_id"]}/{key}',flush=True)
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
            result=publish(out,sample,key,attempt,camera,pixels)
            raw=result[1]['pixel_qa'].get('rawdn',{})
            if max(raw.get('adc_max_fraction',0),raw.get('full_well_or_adc_fraction',0))>.05:
                print(f'QA warning: {sample["sample_id"]}/{key} has >5% RAW at full-well or ADC limit',flush=True)
            return result,attempt
        except (ValueError,OSError,KeyError) as e:
            failed+=1;atomic_json(state,{'status':'failed','error':str(e),'job_sha256':camera['job_sha256'],'attempt':attempt.relative_to(out).as_posix()})
            print(f'FAILED {sample["sample_id"]}/{key}: {e}',flush=True)
            if not args.keep_going:raise _Halt(1)
            return None,None

    try:
        try:
            for sample in selected:
                results={};references={};lwir_attempt=None;halt=None;published=None

                def drop_manifest():
                    stale=out/'samples'/sample['sample_id']/'sample.json'
                    if stale.exists():stale.unlink()
                    update_index(out,plan)

                def drop_references():
                    # The triplet stays published; only its references leave
                    # the manifest until they verify again.
                    sample_manifest(out,plan,sample,results,{});update_index(out,plan)

                try:
                    try:
                        for band in BANDS:
                            result,attempt=acquire(sample,band,drop_manifest)
                            if result:results[band]=result
                            if band=='lwir' and attempt is not None:lwir_attempt=attempt
                        # References follow the triplet; their completeness
                        # never decides the triplet's.
                        for band in sample.get('references',{}):
                            result,_=acquire(sample,REFERENCE_PREFIX+band,drop_references)
                            if result:references[band]=result
                    except _Halt as h:halt=h
                finally:
                    published=sample_manifest(out,plan,sample,results,references)
                    if published:update_index(out,plan)
                if published:
                    print(f'COMPLETE {sample["sample_id"]}',flush=True)
                    completed+=1
                    if audit_every and completed%audit_every==0 and lwir_attempt is not None:
                        # Periodic radiometric closure: recomputing the NN
                        # atmosphere offline and checking L_pred vs corrected
                        # catches mid-run renderer regressions long before
                        # verify does.
                        # Still a standalone script in tools/; resolved from the repository root.
                        audit=Path(__file__).resolve().parents[3]/'tools'/'_closure_audit.py'
                        try:
                            report=run_closure_audit(audit,lwir_attempt)
                            (lwir_attempt/'closure_audit.txt').write_text(report+"\n",encoding='utf-8')
                            summary=[l for l in report.splitlines() if 'closure residual' in l]
                            print(f'CLOSURE {sample["sample_id"]}: {summary[0] if summary else report.splitlines()[-1] if report else "no output"}',flush=True)
                        except Exception as e:
                            print(f'CLOSURE {sample["sample_id"]}: audit unavailable ({e})',flush=True)
                if halt:return halt.code
            print(f'Finished: rendered={captures}, skipped={skipped}, failed={failed}',flush=True)
            return 1 if failed else 0
        finally:update_index(out,plan)
    finally:
        lease['child_pid']=None
