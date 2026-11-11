"""Run/publish/resume/repair/verify contracts against a fake renderer.

The fake fabricates export records that verify_capture accepts, so the whole
acquisition path is exercised without a renderer or GPU.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from quantiloom_fusion.core import jsonio, locking, tomlio
from quantiloom_fusion.datagen import planner, publish, runner, settings, verify

from fake_renderer import FakeRenderer

ROOT = Path(__file__).resolve().parents[1]


class RunDatasetContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cli = self.root/'Quantiloom.exe';self.cli.write_bytes(b'fixture cli')
        (self.root/'assets').mkdir();(self.root/'src/shaders').mkdir(parents=True)
        (self.root/'src/shaders/raygen.hlsl').write_text('// shader v1\n')
        model = self.root/'scene.glb'
        payload = json.dumps({'asset':{'version':'2.0'},'materials':[{'name':'Object'}]}).encode()
        payload += b' ' *((-len(payload))%4)
        model.write_bytes(struct.pack('<III',0x46546c67,2,20+len(payload))+struct.pack('<II',len(payload),0x4e4f534a)+payload)
        (self.root/'solar.csv').write_text('nm,direct,global\n400,1,2\n4000,0.01,0.02\n')
        scene_toml = self.root/'scene.toml'
        scene_toml.write_text(tomlio.toml_text({'renderer':{},'scene':{'gltf':model.as_posix(),'world_units_to_meters':2.0},
            'camera':{'position':[0,1,6],'look_at':[0,1,0]},'lighting':{'solar_lut':'solar.csv'},
            'materials':[{'name':'Object','ir_emissivity':.9}]}))
        self.settings = settings.read_settings(ROOT/'configs/env_dataset.toml')
        self.settings['dataset'].update(cli=self.cli.as_posix(),qt_root=self.root.as_posix(),verification='hashes',
            thermal_mode='authored',ir_materials='constant',geometry_truth=False,daylight_solar_scale='none')
        self.settings['sampling'].update(azimuths_deg=[0,90],elevations_deg=[5],radius_scales=[1.0],repeats=1)
        self.settings['scenes'] = [{'id':'test','config':scene_toml.as_posix()}]
        self.config = self.root/'experiment.toml'
        self.config.write_text(tomlio.toml_text(self.settings))
        self.out = self.root/'dataset'
        self.fake = FakeRenderer()
        self.plan = planner.plan(self.config,self.out,renderer=self.fake)

    def tearDown(self):self.temp.cleanup()

    def drive(self,fake=None,scene=None,max_captures=None,keep_going=False):
        args = argparse.Namespace(scene=scene,max_captures=max_captures,keep_going=keep_going)
        # run_dataset refuses non-drive output dirs on POSIX (the real renderer
        # is Windows-only); the fake needs no such path, so the check is off.
        with patch.object(runner,'POSIX',False),locking.runner_lock(self.out) as (lease,lock):
            return runner.run_dataset(self.out,self.plan,args,lease,lock,renderer=fake or self.fake)

    def test_full_run_indexes_and_verifies(self):
        self.assertEqual(self.drive(),0)
        index_json = jsonio.load_json(self.out/'dataset.index.json')
        self.assertEqual(index_json['complete_triplets'],index_json['planned_triplets'])
        self.assertEqual(verify.verify_dataset(self.out),0)
        for s in self.plan['samples']:
            manifest = jsonio.load_json(self.out/'samples'/s['sample_id']/'sample.json')
            self.assertEqual(set(manifest['observations']),set(self.plan['samples'][0]['cameras']))
            for band,obs in manifest['observations'].items():
                self.assertEqual(set(obs['products']),{'measurement','rawdn','corrected','display','rendered_reference'})
                truth = f'samples/{s["sample_id"]}/{band}/v0001/frame_truth_camera_z.exr'
                self.assertIn(truth,obs['extra_products'])

    def test_resume_is_a_noop(self):
        self.assertEqual(self.drive(),0)
        second = FakeRenderer()
        self.assertEqual(self.drive(fake=second),0)
        self.assertEqual(second.calls,[])

    def test_failure_then_retry_recovers(self):
        sid = self.plan['samples'][0]['sample_id']
        failing = FakeRenderer(exit_codes={(sid,'vis'):[1]})
        self.assertEqual(self.drive(fake=failing),1)
        state = jsonio.load_json(publish.receipt_path(self.out,self.plan['samples'][0],'vis'))
        self.assertEqual(state['status'],'failed')
        self.assertEqual(self.drive(),0)
        attempts = sorted(p.name for p in (self.out/'attempts'/sid/'vis').iterdir())
        self.assertEqual(attempts,['attempt-0001','attempt-0002'])
        self.assertEqual(verify.verify_dataset(self.out),0)

    def test_keep_going_completes_other_samples(self):
        sid = self.plan['samples'][0]['sample_id']
        failing = FakeRenderer(exit_codes={(sid,'vis'):[1]})
        self.assertEqual(self.drive(fake=failing,keep_going=True),1)
        index_json = jsonio.load_json(self.out/'dataset.index.json')
        self.assertEqual(index_json['complete_triplets'],1)
        self.assertNotIn(sid,{s['sample_id'] for s in index_json['samples']})

    def test_corruption_is_repaired(self):
        self.assertEqual(self.drive(),0)
        sid = self.plan['samples'][0]['sample_id']
        product = self.out/'samples'/sid/'vis'/'v0001'/'frame_rawdn.exr'
        data = bytearray(product.read_bytes());data[0] ^= 0xFF
        product.write_bytes(bytes(data))
        with self.assertRaises(ValueError):verify.verify_dataset(self.out)
        repair = FakeRenderer()
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            self.assertEqual(self.drive(fake=repair),0)
        self.assertIn('REPAIR',captured.getvalue())
        self.assertEqual(repair.calls,[(sid,'vis')])
        self.assertEqual(verify.verify_dataset(self.out),0)

    def test_unpublished_attempt_is_recovered(self):
        sid = self.plan['samples'][0]['sample_id']
        camera = self.plan['samples'][0]['cameras']['vis']
        attempt = self.out/'attempts'/sid/'vis'/'attempt-0001'
        attempt.mkdir(parents=True)
        self.fake.write_outputs(self.out/camera['job'],attempt)
        runner_fake = FakeRenderer()
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            self.assertEqual(self.drive(fake=runner_fake),0)
        self.assertIn('RECOVER',captured.getvalue())
        self.assertNotIn((sid,'vis'),runner_fake.calls)
        self.assertEqual(verify.verify_dataset(self.out),0)

    def test_interrupt_marks_receipt_and_releases_child(self):
        sid = self.plan['samples'][0]['sample_id']
        fake = FakeRenderer(interrupt={(sid,'vis')})
        args = argparse.Namespace(scene=None,max_captures=None,keep_going=False)
        with patch.object(runner,'POSIX',False),locking.runner_lock(self.out) as (lease,lock):
            with self.assertRaises(KeyboardInterrupt):
                runner.run_dataset(self.out,self.plan,args,lease,lock,renderer=fake)
            self.assertIn((sid,'vis'),fake.terminated)
            state = jsonio.load_json(publish.receipt_path(self.out,self.plan['samples'][0],'vis'))
            self.assertEqual(state['status'],'interrupted')
            self.assertIsNone(jsonio.load_json(lock)['child_pid'])

    def test_capture_budget_indexes_only_complete_triplets(self):
        fake = FakeRenderer()
        self.assertEqual(self.drive(fake=fake,max_captures=4),0)
        self.assertEqual(len(fake.calls),4)
        index_json = jsonio.load_json(self.out/'dataset.index.json')
        self.assertEqual(index_json['complete_triplets'],1)


if __name__=='__main__':unittest.main()
