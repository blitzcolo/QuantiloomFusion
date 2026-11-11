"""Contract tests for planning, immutability and acquisition recovery.

Set QUANTILOOM_TEST_DATASET to a rendered smoke dataset to exercise real exports.
On WSL export it as a Windows path with WSLENV=QUANTILOOM_TEST_DATASET/p so a
Windows interpreter sees it; the dataset must have been rendered by the account
running the tests (attempt files are owner-ACL'd by the renderer).
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from quantiloom_fusion.core import jsonio, locking, paths, tomlio
from quantiloom_fusion.datagen import constants, export_record, fingerprint, planner, publish, renderer, scene, settings

ROOT=Path(__file__).resolve().parents[1]


class PlanningContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.cli=self.root/'Quantiloom.exe';self.cli.write_bytes(b'fixture cli')
        # A sibling source tree makes the fixture look like a dev checkout:
        # physics_sources then fingerprint these files alongside the version
        # string. Compiled blobs beside the binary stay observed-only.
        (self.root/'assets').mkdir();(self.root/'src/shaders').mkdir(parents=True)
        (self.root/'src/shaders/raygen.hlsl').write_text('// shader v1\n')
        model=self.root/'scene.glb'
        payload=json.dumps({'asset':{'version':'2.0'},'materials':[{'name':'Object'}]}).encode()
        payload+=b' ' *((-len(payload))%4)
        model.write_bytes(struct.pack('<III',0x46546c67,2,20+len(payload))+struct.pack('<II',len(payload),0x4e4f534a)+payload)
        (self.root/'solar.csv').write_text('nm,direct,global\n400,1,2\n4000,0.01,0.02\n')
        self.scene=self.root/'scene.toml'
        self.scene.write_text(tomlio.toml_text({'renderer':{},'scene':{'gltf':model.as_posix(),'world_units_to_meters':2.0},
            'camera':{'position':[0,1,6],'look_at':[0,1,0]},'lighting':{'solar_lut':'solar.csv'},
            'materials':[{'name':'Object','ir_emissivity':.9}]}))
        self.settings=settings.read_settings(ROOT/'configs/env_dataset.toml')
        # The fixture scene carries no [thermal] or spectral bindings, so the
        # formal config's solved/spectral policies cannot run against it.
        self.settings['dataset'].update(cli=self.cli.as_posix(),qt_root=self.root.as_posix(),verification='hashes',
            thermal_mode='authored',ir_materials='constant',geometry_truth=False,daylight_solar_scale='none')
        self.settings['sampling'].update(azimuths_deg=[0,90,180],elevations_deg=[5,25],radius_scales=[.8,1.2],repeats=2)
        self.settings['scenes']=[{'id':'test','config':self.scene.as_posix()}]
        self.config=self.root/'experiment.toml';self.save()
        self.out=self.root/'dataset'

    def tearDown(self):self.temp.cleanup()
    def save(self):self.config.write_text(tomlio.toml_text(self.settings))
    def plan(self):
        with patch.object(renderer.subprocess,'check_output',return_value='Quantiloom 0.7.0\n'):
            return planner.plan(self.config,self.out)

    def test_cartesian_coverage_fixed_profiles_and_metric_baselines(self):
        plan=self.plan()
        self.assertEqual(len(plan['samples']),24)
        self.assertEqual(len({s['sample_id'] for s in plan['samples']}),24)
        profiles={}
        for sample in plan['samples']:
            self.assertEqual(set(sample['cameras']),set(constants.BANDS))
            poses={b:c['camera_to_world'] for b,c in sample['cameras'].items()}
            for b,c in sample['cameras'].items():
                job=tomllib.loads((self.out/c['job']).read_text())
                self.assertEqual(job['camera']['fov_y'],self.settings['cameras'][b]['fov_y_deg'])
                self.assertEqual(job['renderer']['resolution'],self.settings['cameras'][b]['resolution'])
                # noise_seed legitimately varies per sample under per_sample policy
                sensor={k:v for k,v in job['sensor'].items() if k!='noise_seed'}
                if b in profiles:self.assertEqual(sensor,profiles[b])
                profiles[b]=sensor
            for a,b in [('vis','mwir'),('mwir','lwir'),('vis','lwir')]:
                distance=2*math.dist([poses[a][i] for i in (3,7,11)],[poses[b][i] for i in (3,7,11)])
                expected=math.dist(self.settings['cameras'][a]['offset_rdf_m'],self.settings['cameras'][b]['offset_rdf_m'])
                self.assertAlmostEqual(distance,expected,places=10)
        self.assertEqual(len({s['split_group'] for s in plan['samples']}),1)

    def test_source_change_rejected_without_overwriting_derived_assets(self):
        before=self.plan();solar=self.out/'provenance/solar_test.csv';original=solar.read_bytes()
        (self.root/'solar.csv').write_text('nm,direct,global\n400,1,2\n4000,0.04,0.08\n')
        with self.assertRaisesRegex(ValueError,'different recipe'):self.plan()
        self.assertEqual(solar.read_bytes(),original)
        self.assertEqual(jsonio.load_json(self.out/'dataset.plan.json')['dataset_id'],before['dataset_id'])

    def test_job_edit_and_parameter_typo_rejected(self):
        p=self.plan();job=self.out/p['samples'][0]['cameras']['vis']['job']
        job.write_text(job.read_text()+'# edited\n')
        with self.assertRaisesRegex(ValueError,'planned file changed'):self.plan()
        self.settings['cameras']['lwir']['responsivity_dn_per_w']=1e12;self.save()
        with self.assertRaisesRegex(ValueError,'unknown cameras.lwir'):settings.read_settings(self.config)

    def test_replan_keeps_stored_plan_across_ulp_differences(self):
        # libm differs in the last ULP between OSes: the first planning host
        # freezes the plan, and a tolerant re-plan leaves every byte alone.
        self.plan();cam='lwir'
        stored=jsonio.load_json(self.out/'dataset.plan.json');camera=stored['samples'][0]['cameras'][cam]
        job=self.out/camera['job'];doc=tomllib.loads(job.read_text())
        doc['camera']['position'][0]=math.nextafter(doc['camera']['position'][0],math.inf)
        job.write_text(tomlio.toml_text(doc))
        camera['job_sha256']=hashlib.sha256(job.read_bytes()).hexdigest()
        camera['camera_to_world'][3]=doc['camera']['position'][0]
        jsonio.atomic_json(self.out/'dataset.plan.json',stored)
        before={p.relative_to(self.out):p.read_bytes() for p in self.out.rglob('*') if p.is_file()}
        second=self.plan()
        after={p.relative_to(self.out):p.read_bytes() for p in self.out.rglob('*') if p.is_file()}
        self.assertEqual(after,before)
        self.assertEqual(second['samples'][0]['cameras'][cam]['job_sha256'],camera['job_sha256'])

    def test_replan_rejects_real_job_change(self):
        self.plan();cam='vis'
        stored=jsonio.load_json(self.out/'dataset.plan.json');camera=stored['samples'][0]['cameras'][cam]
        job=self.out/camera['job'];doc=tomllib.loads(job.read_text())
        doc['renderer']['spp']=doc['renderer'].get('spp',1)+1
        job.write_text(tomlio.toml_text(doc))
        camera['job_sha256']=hashlib.sha256(job.read_bytes()).hexdigest()
        jsonio.atomic_json(self.out/'dataset.plan.json',stored)
        with self.assertRaisesRegex(ValueError,'planned file changed'):self.plan()

    def test_replan_rejects_unrecorded_job_edit(self):
        # A within-tolerance edit still fails because the file no longer hashes
        # to the plan's job_sha256.
        self.plan();cam='mwir'
        camera=jsonio.load_json(self.out/'dataset.plan.json')['samples'][0]['cameras'][cam]
        job=self.out/camera['job'];doc=tomllib.loads(job.read_text())
        doc['camera']['fov_y']=math.nextafter(doc['camera']['fov_y'],math.inf)
        job.write_text(tomlio.toml_text(doc))
        with self.assertRaisesRegex(ValueError,'planned file changed'):self.plan()

    def test_shader_change_accepted_within_same_cli_version(self):
        # Compiled blobs beside the binary are recorded in renderer_observed
        # but do not gate: relinking identical sources would churn otherwise.
        # Their *sources* gate instead -- see the physics_sources contract.
        shader=self.root/'raygen.spv';shader.write_bytes(b'compiled shader A')
        first=self.plan();shader.write_bytes(b'compiled shader B')
        second=self.plan()
        self.assertEqual(first['dataset_id'],second['dataset_id'])
        observed=jsonio.load_json(self.out/'dataset.plan.json')['renderer_observed']
        self.assertIn('raygen.spv',{Path(e['path']).name for e in observed})
        with patch.object(renderer.subprocess,'check_output',return_value='Quantiloom 0.7.1\n'):
            with self.assertRaisesRegex(ValueError,'different recipe'):planner.plan(self.config,self.out)

    def test_physics_source_change_rejected_within_same_cli_version(self):
        # A same-version rebuild that edits physics-relevant sources must not
        # resume a dataset: that is exactly the pre/post-fix mixing the
        # version string alone could not stop.
        self.plan()
        recipe=jsonio.load_json(self.out/'dataset.plan.json')['recipe']
        self.assertTrue(recipe['physics_sources'])
        self.assertIn('quantiloom_fusion/datagen/planner.py',{e['path'] for e in recipe['generator_files']})
        self.assertEqual(recipe['generator_sha256'],fingerprint.generator_fingerprint(recipe['generator_files']))
        (self.root/'src/shaders/raygen.hlsl').write_text('// shader v2\n')
        with self.assertRaisesRegex(ValueError,'different recipe'):self.plan()

    def test_per_acquisition_noise_seed_and_scene_overrides(self):
        plan=self.plan()
        by_band={};acq=set()
        for s in plan['samples']:
            for b in constants.BANDS:
                job=tomllib.loads((self.out/s['cameras'][b]['job']).read_text())
                # per_acquisition: one detector seed per band (FPN maps hold
                # still), a distinct acquisition_index per capture drives the
                # temporal noise classes.
                by_band.setdefault(b,set()).add(job['sensor']['noise_seed'])
                acq.add(job['dataset']['acquisition_index'])
        self.assertTrue(all(len(v)==1 for v in by_band.values()))
        self.assertEqual(len(acq),len(plan['samples'])*len(constants.BANDS))
        # fixed_detector: one seed per band and no acquisition counter. Policy
        # switches change the recipe, so each branch plans into its own
        # directory.
        self.settings['dataset']['noise_seed_policy']='fixed_detector';self.save()
        out2=self.root/'dataset2'
        with patch.object(renderer.subprocess,'check_output',return_value='Quantiloom 0.7.0\n'):
            plan2=planner.plan(self.config,out2)
        by_band={}
        for s in plan2['samples']:
            for b in constants.BANDS:
                job=tomllib.loads((out2/s['cameras'][b]['job']).read_text())
                by_band.setdefault(b,set()).add(job['sensor']['noise_seed'])
                self.assertNotIn('acquisition_index',job['dataset'])
        self.assertTrue(all(len(v)==1 for v in by_band.values()))
        # per_sample: every capture draws all noise classes fresh.
        self.settings['dataset']['noise_seed_policy']='per_sample';self.save()
        out4=self.root/'dataset4'
        with patch.object(renderer.subprocess,'check_output',return_value='Quantiloom 0.7.0\n'):
            plan4=planner.plan(self.config,out4)
        seeds={tomllib.loads((out4/s['cameras'][b]['job']).read_text())['sensor']['noise_seed']
               for s in plan4['samples'] for b in constants.BANDS}
        self.assertEqual(len(seeds),len(plan4['samples'])*len(constants.BANDS))
        # overrides replace whole named sections of the scene config
        self.settings['scenes'][0]['overrides']={'lighting':{'sun_radiance':[0.0,0.0,0.0],'sky_radiance':[0.0,0.0,0.0],'sun_direction':[0.0,1.0,0.0]}};self.save()
        out3=self.root/'dataset3'
        with patch.object(renderer.subprocess,'check_output',return_value='Quantiloom 0.7.0\n'):
            plan3=planner.plan(self.config,out3)
        job=tomllib.loads((out3/plan3['samples'][0]['cameras']['lwir']['job']).read_text())
        self.assertEqual(job['lighting'],plan3['scenes'][0]['overrides']['lighting'])
        # solar_lut absent from a wholesale-replaced section -> no derived CSV
        self.assertEqual(plan3['scenes'][0]['assumptions']['ir_solar_tail'],'source CSV unchanged')

    def test_daylight_solar_scale_matches_forcing_beam(self):
        # LUT direct column: (4000-400)*(1+0.01)/2 = 1818 W/m2; forcing beam at
        # 12 h interpolates 500 and 900 to 700 W/m2.
        (self.root/'day.csv').write_text('# preamble\n#time_h,air,sun\n10.0,290,500.0,0,30,270,50,40\n14.0,300,900.0,0,50,280,60,30\n')
        doc=tomllib.loads(self.scene.read_text())
        doc['lighting']['solar_lut_columns']=[2,3]
        doc['thermal']={'enabled':True,'time_h':12.0,'forcing_file':'day.csv'}
        self.scene.write_text(tomlio.toml_text(doc))
        self.settings['dataset']['daylight_solar_scale']='forcing_direct'
        night={'lighting':{'sun_direction':[0,-1,0]},'thermal':{'enabled':True,'time_h':23.0,'forcing_file':'day.csv'}}
        self.settings['scenes']=[{'id':'test','config':self.scene.as_posix()},
            {'id':'test_low','config':self.scene.as_posix(),'solar_scale':2e-6},
            {'id':'test_night','config':self.scene.as_posix(),'overrides':night}]
        self.save();p=self.plan()
        info={s['id']:s for s in p['scenes']}
        ratio=700/1818
        day=info['test']['assumptions']
        self.assertAlmostEqual(day['solar_scale'],ratio,places=12)
        self.assertEqual({k:day['solar_scale_derivation'][k] for k in ('policy','time_h','reference_column','arm_scale')},
                         {'policy':'forcing_direct','time_h':12.0,'reference_column':2,'arm_scale':1.0})
        self.assertAlmostEqual(day['solar_scale_derivation']['forcing_direct_w_m2'],700.0,places=9)
        self.assertAlmostEqual(day['solar_scale_derivation']['reference_direct_w_m2'],1818.0,places=9)
        # The low-light factor multiplies the forcing match; the night arm has no sun.
        self.assertAlmostEqual(info['test_low']['assumptions']['solar_scale'],2e-6*ratio,delta=1e-18)
        self.assertEqual(info['test_night']['assumptions']['solar_scale'],1.0)
        rows=(self.out/'provenance/solar_test.csv').read_text().splitlines()
        self.assertEqual(rows[0],'nm,direct,global')  # the renderer reads column labels
        self.assertAlmostEqual(float(rows[1].split(',')[1]),ratio,places=11)
        low=(self.out/'provenance/solar_test_low.csv').read_text().splitlines()
        self.assertAlmostEqual(float(low[1].split(',')[1]),2e-6*ratio,delta=1e-17)
        self.assertFalse((self.out/'provenance/solar_test_night.csv').exists())
        # A lit scene without a forcing file cannot be matched to one.
        del doc['thermal'];self.scene.write_text(tomlio.toml_text(doc))
        self.out=self.root/'dataset2'
        with self.assertRaisesRegex(ValueError,'forcing_file and time_h'):self.plan()

    def test_fixed_pattern_and_nuc_reach_the_job_and_are_validated(self):
        p=self.plan();cams=p['samples'][0]['cameras']
        job={b:tomllib.loads((self.out/cams[b]['job']).read_text())['sensor'] for b in constants.BANDS}
        mw=job['mwir']['photon']
        self.assertEqual({k:mw[k] for k in ('prnu_sigma','dsnu_e_rms','column_gain_sigma','column_offset_e_rms','row_gain_sigma','row_offset_e_rms','apply_nuc','nuc_residual_fraction')},
                         {'prnu_sigma':0.0074,'dsnu_e_rms':5.9,'column_gain_sigma':0.006,'column_offset_e_rms':4.8,'row_gain_sigma':0.003,'row_offset_e_rms':2.4,'apply_nuc':True,'nuc_residual_fraction':0.03})
        self.assertEqual((job['vis']['photon']['apply_nuc'],job['vis']['photon']['column_gain_sigma']),(False,0.0))
        lw=job['lwir']['thermal']
        self.assertEqual({k:lw[k] for k in ('enable_fpn','fpn_gain_sigma','fpn_offset_k','fpn_column_gain_sigma','fpn_column_offset_k','fpn_row_gain_sigma','fpn_row_offset_k','apply_nuc','nuc_residual_fraction')},
                         {'enable_fpn':True,'fpn_gain_sigma':0.016,'fpn_offset_k':1.1,'fpn_column_gain_sigma':0.023,'fpn_column_offset_k':1.55,'fpn_row_gain_sigma':0.01,'fpn_row_offset_k':0.65,'apply_nuc':True,'nuc_residual_fraction':0.03})
        # Each component's variance adds; the totals stay at 1 %/8 e- and 3 %/2 K.
        self.assertAlmostEqual(math.hypot(mw['prnu_sigma'],mw['column_gain_sigma'],mw['row_gain_sigma']),0.01,places=4)
        self.assertAlmostEqual(math.hypot(mw['dsnu_e_rms'],mw['column_offset_e_rms'],mw['row_offset_e_rms']),8.0,places=1)
        self.assertAlmostEqual(math.hypot(lw['fpn_gain_sigma'],lw['fpn_column_gain_sigma'],lw['fpn_row_gain_sigma']),0.03,places=3)
        self.assertAlmostEqual(math.hypot(lw['fpn_offset_k'],lw['fpn_column_offset_k'],lw['fpn_row_offset_k']),2.0,delta=0.01)
        cases=[('mwir','nuc_residual_fraction',1.5,r'cameras\.mwir\.nuc_residual_fraction must be a number in \[0, 1\]'),
               ('mwir','apply_nuc',1,r'cameras\.mwir\.apply_nuc must be true or false'),
               ('lwir','fpn_offset_k',-1.0,r'cameras\.lwir\.fpn_offset_k must be a finite number >= 0'),
               ('vis','row_offset_e_rms','x',r'cameras\.vis\.row_offset_e_rms must be a finite number >= 0'),
               ('lwir','fpn_column_gain_sigma',1.2,r'cameras\.lwir\.fpn_column_gain_sigma is a relative gain deviation'),
               ('lwir','column_gain_sigma',0.01,r'cameras\.lwir\.column_gain_sigma belongs to the other detector kind'),
               ('mwir','column_offset_e_rms',None,r'cameras\.mwir\.column_offset_e_rms is required')]
        for band,key,value,message in cases:
            with self.subTest(band=band,key=key):
                broken=copy.deepcopy(self.settings)
                if value is None:del broken['cameras'][band][key]
                else:broken['cameras'][band][key]=value
                self.config.write_text(tomlio.toml_text(broken))
                with self.assertRaisesRegex(ValueError,message):settings.read_settings(self.config)

    def test_pose_survives_float32_far_from_origin(self):
        # The renderer rebuilds the view basis in float32 from look_at - position.
        self.settings['scenes'][0].update(target=[-508.6,-117.6,-406.4],radius_m=600.0);self.save()
        plan=self.plan()
        f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
        for s in plan['samples']:
            for b in constants.BANDS:
                camera=tomllib.loads((self.out/s['cameras'][b]['job']).read_text())['camera']
                d=[f32(f32(l)-f32(p)) for l,p in zip(camera['look_at'],camera['position'])]
                n=math.sqrt(sum(x*x for x in d))
                c2w=s['cameras'][b]['camera_to_world']
                for got,want in zip(d,(c2w[2],c2w[6],c2w[10])):
                    self.assertLess(abs(got/n-want),2e-6)

    def test_one_humidity_key_per_job(self):
        d={'spp':1,'geometry_truth':False,'atmosphere_mode':'source','thermal_mode':'authored','ir_materials':'constant'}
        base={'atmosphere':{'preset':'clear','rh':0.75,'relative_humidity':50},'materials':[]}
        for band in constants.BANDS:
            self.assertEqual(scene.scene_for_band(base,band,[],d,None)['atmosphere'],{'preset':'clear','rh':0.75})
        base['atmosphere']={'preset':'clear','relative_humidity':55}
        self.assertEqual(scene.scene_for_band(base,'lwir',[],d,None)['atmosphere'],{'preset':'clear','relative_humidity':55})

    def test_nested_motion_and_duplicate_camera_offsets_rejected(self):
        with self.assertRaisesRegex(ValueError,'model motion'):scene.reject_motion({'models':[{'motion':{'velocity':[1,0,0]}}]})
        self.settings['cameras']['mwir']['offset_rdf_m']=self.settings['cameras']['vis']['offset_rdf_m'];self.save()
        with self.assertRaisesRegex(ValueError,'distinct local positions'):settings.read_settings(self.config)

    def test_path_translation_round_trip(self):
        self.assertEqual(paths.native_path('H:/a/b',posix=True),Path('/mnt/h/a/b'))
        self.assertEqual(paths.native_path('h:\\a\\b',posix=True),Path('/mnt/h/a/b'))
        self.assertEqual(paths.native_path('rel/c.toml',posix=True),Path('rel/c.toml'))
        self.assertEqual(paths.native_path('/mnt/h/a',posix=False),Path('H:/a'))
        self.assertEqual(paths.canonical_path(Path('/mnt/h/a'),posix=True),'H:/a')
        self.assertEqual(paths.canonical_path(Path('H:/a'),posix=False),'H:/a')
        self.assertEqual(paths.canonical_path(Path('/tmp/x'),posix=True),'/tmp/x')

    def test_single_writer_and_crash_released_mutex(self):
        self.out.mkdir()
        with locking.runner_lock(self.out):
            with self.assertRaisesRegex(ValueError,'active'):
                with locking.runner_lock(self.out):pass
        jsonio.atomic_json(self.out/'.generator.lock',{'host':socket.gethostname(),'pid':-1,'child_pid':None,'token':'stale'})
        with locking.runner_lock(self.out):pass
        self.assertFalse((self.out/'.generator.lock').exists())

    def test_tools_shim_runs_help(self):
        env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
        with tempfile.TemporaryDirectory() as cwd:
            proc=subprocess.run([sys.executable,str(ROOT/'tools/generate_env_dataset.py'),'--help'],
                                cwd=cwd,env=env,capture_output=True,text=True)
        self.assertEqual(proc.returncode,0)
        self.assertIn('plan',proc.stdout)
        self.assertIn('verify',proc.stdout)


@unittest.skipUnless(os.environ.get('QUANTILOOM_TEST_DATASET'),'set QUANTILOOM_TEST_DATASET for real export tests')
class ExportRecoveryContracts(unittest.TestCase):
    def setUp(self):
        self.source=Path(os.environ['QUANTILOOM_TEST_DATASET']).resolve()
        self.plan=jsonio.load_json(self.source/'dataset.plan.json');self.sample=self.plan['samples'][0]
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)
        self.band='vis';self.camera=self.sample['cameras'][self.band]
        self.attempt=self.source/'attempts'/self.sample['sample_id']/self.band/'attempt-0001'

    def tearDown(self):self.temp.cleanup()

    def test_recover_published_files_without_receipt_and_reject_corruption(self):
        receipt,summary=publish.publish(self.out,self.sample,self.band,self.attempt,self.camera,True)
        publish.receipt_path(self.out,self.sample,self.band).unlink()
        recovered,_=publish.publish(self.out,self.sample,self.band,self.attempt,self.camera,True)
        self.assertEqual(recovered['record_id'],receipt['record_id'])
        self.assertIsNotNone(publish.valid_receipt(self.out,self.sample,self.band,True))
        product=self.out/Path(receipt['record']).parent/'frame_rawdn.exr'
        with product.open('r+b') as f:f.seek(16);f.write(b'CORRUPT!')
        with self.assertRaisesRegex(ValueError,'corrupt product'):publish.valid_receipt(self.out,self.sample,self.band,False)

    def test_effective_pose_and_fov_differences_rejected(self):
        wrong=copy.deepcopy(self.camera);wrong['camera_to_world'][3]+=.02
        with self.assertRaisesRegex(ValueError,'camera pose'):export_record.verify_capture(self.attempt/'frame.metadata.json',wrong,False)
        wrong=copy.deepcopy(self.camera);wrong['profile']['fov_y_deg']+=1
        with self.assertRaisesRegex(ValueError,'FOV/intrinsics'):export_record.verify_capture(self.attempt/'frame.metadata.json',wrong,False)


if __name__=='__main__':unittest.main()
