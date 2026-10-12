"""Contract tests for planning, immutability and acquisition recovery.

Set QUANTILOOM_TEST_DATASET to a rendered smoke dataset to exercise real exports.
On WSL export it as a Windows path with WSLENV=QUANTILOOM_TEST_DATASET/p so a
Windows interpreter sees it; the dataset must have been rendered by the account
running the tests (attempt files are owner-ACL'd by the renderer).
"""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'tools/generate_env_dataset.py'
spec=importlib.util.spec_from_file_location('generator',SCRIPT)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
ROOT=SCRIPT.parents[1]


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
        self.scene.write_text(g.toml_text({'renderer':{},'scene':{'gltf':model.as_posix(),'world_units_to_meters':2.0},
            'camera':{'position':[0,1,6],'look_at':[0,1,0]},'lighting':{'solar_lut':'solar.csv'},
            'materials':[{'name':'Object','ir_emissivity':.9}]}))
        self.settings=g.read_settings(ROOT/'configs/env_dataset.toml')
        # The fixture scene carries no [thermal] or spectral bindings, so the
        # formal config's solved/spectral policies cannot run against it.
        self.settings['dataset'].update(cli=self.cli.as_posix(),qt_root=self.root.as_posix(),verification='hashes',
            thermal_mode='authored',ir_materials='constant',geometry_truth=False)
        self.settings['sampling'].update(azimuths_deg=[0,90,180],elevations_deg=[5,25],radius_scales=[.8,1.2],repeats=2)
        self.settings['scenes']=[{'id':'test','config':self.scene.as_posix()}]
        self.config=self.root/'experiment.toml';self.save()
        self.out=self.root/'dataset'

    def tearDown(self):self.temp.cleanup()
    def save(self):self.config.write_text(g.toml_text(self.settings))
    def plan(self):
        with patch.object(g.subprocess,'check_output',return_value='Quantiloom 0.7.0\n'):
            return g.plan(self.config,self.out)

    def test_cartesian_coverage_fixed_profiles_and_metric_baselines(self):
        plan=self.plan()
        self.assertEqual(len(plan['samples']),24)
        self.assertEqual(len({s['sample_id'] for s in plan['samples']}),24)
        profiles={}
        for sample in plan['samples']:
            self.assertEqual(set(sample['cameras']),set(g.BANDS))
            poses={b:c['camera_to_world'] for b,c in sample['cameras'].items()}
            for b,c in sample['cameras'].items():
                job=g.tomllib.loads((self.out/c['job']).read_text())
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
        self.assertEqual(g.load_json(self.out/'dataset.plan.json')['dataset_id'],before['dataset_id'])

    def test_job_edit_and_parameter_typo_rejected(self):
        p=self.plan();job=self.out/p['samples'][0]['cameras']['vis']['job']
        job.write_text(job.read_text()+'# edited\n')
        with self.assertRaisesRegex(ValueError,'planned file changed'):self.plan()
        self.settings['cameras']['lwir']['responsivity_dn_per_w']=1e12;self.save()
        with self.assertRaisesRegex(ValueError,'unknown cameras.lwir'):g.read_settings(self.config)

    def test_shader_change_accepted_within_same_cli_version(self):
        # Compiled blobs beside the binary are recorded in renderer_observed
        # but do not gate: relinking identical sources would churn otherwise.
        # Their *sources* gate instead -- see the physics_sources contract.
        shader=self.root/'raygen.spv';shader.write_bytes(b'compiled shader A')
        first=self.plan();shader.write_bytes(b'compiled shader B')
        second=self.plan()
        self.assertEqual(first['dataset_id'],second['dataset_id'])
        observed=g.load_json(self.out/'dataset.plan.json')['renderer_observed']
        self.assertIn('raygen.spv',{Path(e['path']).name for e in observed})
        with patch.object(g.subprocess,'check_output',return_value='Quantiloom 0.7.1\n'):
            with self.assertRaisesRegex(ValueError,'different recipe'):g.plan(self.config,self.out)

    def test_physics_source_change_rejected_within_same_cli_version(self):
        # A same-version rebuild that edits physics-relevant sources must not
        # resume a dataset: that is exactly the pre/post-fix mixing the
        # version string alone could not stop.
        self.plan()
        self.assertTrue(g.load_json(self.out/'dataset.plan.json')['recipe']['physics_sources'])
        (self.root/'src/shaders/raygen.hlsl').write_text('// shader v2\n')
        with self.assertRaisesRegex(ValueError,'different recipe'):self.plan()

    def test_per_acquisition_noise_seed_and_scene_overrides(self):
        plan=self.plan()
        by_band={};acq=set()
        for s in plan['samples']:
            for b in g.BANDS:
                job=g.tomllib.loads((self.out/s['cameras'][b]['job']).read_text())
                # per_acquisition: one detector seed per band (FPN maps hold
                # still), a distinct acquisition_index per capture drives the
                # temporal noise classes.
                by_band.setdefault(b,set()).add(job['sensor']['noise_seed'])
                acq.add(job['dataset']['acquisition_index'])
        self.assertTrue(all(len(v)==1 for v in by_band.values()))
        self.assertEqual(len(acq),len(plan['samples'])*len(g.BANDS))
        # fixed_detector: one seed per band and no acquisition counter. Policy
        # switches change the recipe, so each branch plans into its own
        # directory.
        self.settings['dataset']['noise_seed_policy']='fixed_detector';self.save()
        out2=self.root/'dataset2'
        with patch.object(g.subprocess,'check_output',return_value='Quantiloom 0.7.0\n'):
            plan2=g.plan(self.config,out2)
        by_band={}
        for s in plan2['samples']:
            for b in g.BANDS:
                job=g.tomllib.loads((out2/s['cameras'][b]['job']).read_text())
                by_band.setdefault(b,set()).add(job['sensor']['noise_seed'])
                self.assertNotIn('acquisition_index',job['dataset'])
        self.assertTrue(all(len(v)==1 for v in by_band.values()))
        # per_sample: every capture draws all noise classes fresh.
        self.settings['dataset']['noise_seed_policy']='per_sample';self.save()
        out4=self.root/'dataset4'
        with patch.object(g.subprocess,'check_output',return_value='Quantiloom 0.7.0\n'):
            plan4=g.plan(self.config,out4)
        seeds={g.tomllib.loads((out4/s['cameras'][b]['job']).read_text())['sensor']['noise_seed']
               for s in plan4['samples'] for b in g.BANDS}
        self.assertEqual(len(seeds),len(plan4['samples'])*len(g.BANDS))
        # overrides replace whole named sections of the scene config
        self.settings['scenes'][0]['overrides']={'lighting':{'sun_radiance':[0.0,0.0,0.0],'sky_radiance':[0.0,0.0,0.0],'sun_direction':[0.0,1.0,0.0]}};self.save()
        out3=self.root/'dataset3'
        with patch.object(g.subprocess,'check_output',return_value='Quantiloom 0.7.0\n'):
            plan3=g.plan(self.config,out3)
        job=g.tomllib.loads((out3/plan3['samples'][0]['cameras']['lwir']['job']).read_text())
        self.assertEqual(job['lighting'],plan3['scenes'][0]['overrides']['lighting'])
        # solar_lut absent from a wholesale-replaced section -> no derived CSV
        self.assertEqual(plan3['scenes'][0]['assumptions']['ir_solar_tail'],'source CSV unchanged')

    def test_nested_motion_and_duplicate_camera_offsets_rejected(self):
        with self.assertRaisesRegex(ValueError,'model motion'):g.reject_motion({'models':[{'motion':{'velocity':[1,0,0]}}]})
        self.settings['cameras']['mwir']['offset_rdf_m']=self.settings['cameras']['vis']['offset_rdf_m'];self.save()
        with self.assertRaisesRegex(ValueError,'distinct local positions'):g.read_settings(self.config)

    def test_path_translation_round_trip(self):
        self.assertEqual(g.native_path('H:/a/b',posix=True),Path('/mnt/h/a/b'))
        self.assertEqual(g.native_path('h:\\a\\b',posix=True),Path('/mnt/h/a/b'))
        self.assertEqual(g.native_path('rel/c.toml',posix=True),Path('rel/c.toml'))
        self.assertEqual(g.native_path('/mnt/h/a',posix=False),Path('H:/a'))
        self.assertEqual(g.canonical_path(Path('/mnt/h/a'),posix=True),'H:/a')
        self.assertEqual(g.canonical_path(Path('H:/a'),posix=False),'H:/a')
        self.assertEqual(g.canonical_path(Path('/tmp/x'),posix=True),'/tmp/x')

    def test_single_writer_and_crash_released_mutex(self):
        self.out.mkdir()
        with g.runner_lock(self.out):
            with self.assertRaisesRegex(ValueError,'active'):
                with g.runner_lock(self.out):pass
        g.atomic_json(self.out/'.generator.lock',{'host':g.socket.gethostname(),'pid':-1,'child_pid':None,'token':'stale'})
        with g.runner_lock(self.out):pass
        self.assertFalse((self.out/'.generator.lock').exists())


@unittest.skipUnless(os.environ.get('QUANTILOOM_TEST_DATASET'),'set QUANTILOOM_TEST_DATASET for real export tests')
class ExportRecoveryContracts(unittest.TestCase):
    def setUp(self):
        self.source=Path(os.environ['QUANTILOOM_TEST_DATASET']).resolve()
        self.plan=g.load_json(self.source/'dataset.plan.json');self.sample=self.plan['samples'][0]
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)
        self.band='vis';self.camera=self.sample['cameras'][self.band]
        self.attempt=self.source/'attempts'/self.sample['sample_id']/self.band/'attempt-0001'

    def tearDown(self):self.temp.cleanup()

    def test_recover_published_files_without_receipt_and_reject_corruption(self):
        receipt,summary=g.publish(self.out,self.sample,self.band,self.attempt,self.camera,True)
        g.receipt_path(self.out,self.sample,self.band).unlink()
        recovered,_=g.publish(self.out,self.sample,self.band,self.attempt,self.camera,True)
        self.assertEqual(recovered['record_id'],receipt['record_id'])
        self.assertIsNotNone(g.valid_receipt(self.out,self.sample,self.band,True))
        product=self.out/Path(receipt['record']).parent/'frame_rawdn.exr'
        with product.open('r+b') as f:f.seek(16);f.write(b'CORRUPT!')
        with self.assertRaisesRegex(ValueError,'corrupt product'):g.valid_receipt(self.out,self.sample,self.band,False)

    def test_effective_pose_and_fov_differences_rejected(self):
        wrong=copy.deepcopy(self.camera);wrong['camera_to_world'][3]+=.02
        with self.assertRaisesRegex(ValueError,'camera pose'):g.verify_capture(self.attempt/'frame.metadata.json',wrong,False)
        wrong=copy.deepcopy(self.camera);wrong['profile']['fov_y_deg']+=1
        with self.assertRaisesRegex(ValueError,'FOV/intrinsics'):g.verify_capture(self.attempt/'frame.metadata.json',wrong,False)


if __name__=='__main__':unittest.main()
