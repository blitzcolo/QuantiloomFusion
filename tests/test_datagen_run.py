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
import tomllib
import unittest
from unittest.mock import patch

from quantiloom_fusion.core import jsonio, locking, tomlio
from quantiloom_fusion.datagen import constants, planner, publish, runner, settings, verify

from fake_renderer import FakeRenderer

ROOT = Path(__file__).resolve().parents[1]


class _PlannedFixture(unittest.TestCase):
    subset = None  # [reference_subset] table, or None for no references

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
        self.settings.pop('reference_subset')
        if self.subset:self.settings['reference_subset'] = self.subset
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

    def verify_output(self):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):self.assertEqual(verify.verify_dataset(self.out),0)
        return captured.getvalue()


class RunDatasetContracts(_PlannedFixture):
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


class ReferenceRunContracts(_PlannedFixture):
    # Of the two rig cells (azimuth 0 and 90) only azimuth 0 gets references.
    subset = {'azimuths_deg':[0],'elevations_deg':[5],'radius_scales':[1.0]}

    def setUp(self):
        super().setUp()
        self.ref,self.plain = self.plan['samples']
        self.sid = self.ref['sample_id']

    def manifest(self,sample):return jsonio.load_json(self.out/'samples'/sample['sample_id']/'sample.json')
    def index_flags(self):return [e['references'] for e in jsonio.load_json(self.out/'dataset.index.json')['samples']]

    def test_references_follow_the_triplet_and_publish_in_sample_v2(self):
        self.assertEqual(self.drive(),0)
        self.assertEqual([k for s,k in self.fake.calls if s==self.sid],['vis','mwir','lwir','ref_vis','ref_mwir','ref_lwir'])
        self.assertEqual([k for s,k in self.fake.calls if s==self.plain['sample_id']],['vis','mwir','lwir'])
        m = self.manifest(self.ref)
        self.assertEqual((m['schema_version'],set(m['references'])),(2,set(constants.BANDS)))
        vis_geometry = m['observations']['vis']['products']['measurement']['provenance']['geometry']
        for band,ref in m['references'].items():
            self.assertEqual((ref['geometry_from'],ref['psf_sigma_px']),('vis',self.settings['cameras']['vis']['psf_sigma_px']))
            self.assertEqual(set(ref['products']),set(m['observations'][band]['products']))
            self.assertEqual(Path(ref['record']).parent.as_posix(),f'samples/{self.sid}/ref_{band}/v0001')
            provenance = ref['products']['measurement']['provenance']
            self.assertEqual(provenance['geometry'],vis_geometry)
            sensor = tomllib.loads(provenance['camera_config_toml'])['sensor']
            real = tomllib.loads(m['observations'][band]['products']['measurement']['provenance']['camera_config_toml'])['sensor']
            self.assertEqual((sensor['quality']['noise_free'],sensor['preset']),(True,real['preset']))
            receipt = jsonio.load_json(self.out/'state'/self.sid/f'ref_{band}.json')
            self.assertEqual((receipt['status'],receipt['record']),('complete',ref['record']))
        self.assertEqual(self.manifest(self.plain)['references'],{})
        self.assertEqual(self.index_flags(),[True,False])
        self.assertIn('references=1/1',self.verify_output())

    def test_resume_is_a_noop(self):
        self.assertEqual(self.drive(),0)
        second = FakeRenderer()
        self.assertEqual(self.drive(fake=second),0)
        self.assertEqual(second.calls,[])
        self.assertEqual(self.index_flags(),[True,False])

    def test_corrupted_reference_is_detected_and_repaired(self):
        self.assertEqual(self.drive(),0)
        product = self.out/'samples'/self.sid/'ref_mwir'/'v0001'/'frame_rawdn.exr'
        data = bytearray(product.read_bytes());data[0] ^= 0xFF
        product.write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError,'corrupt product'):verify.verify_dataset(self.out)
        repair = FakeRenderer()
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):self.assertEqual(self.drive(fake=repair),0)
        self.assertIn(f'REPAIR {self.sid}/ref_mwir',captured.getvalue())
        self.assertEqual(repair.calls,[(self.sid,'ref_mwir')])
        self.assertEqual(sorted(p.name for p in (self.out/'attempts'/self.sid/'ref_mwir').iterdir()),['attempt-0001','attempt-0002'])
        self.assertEqual(Path(self.manifest(self.ref)['references']['mwir']['record']).parent.name,'v0002')
        self.assertIn('references=1/1',self.verify_output())

    def test_capture_budget_counts_references(self):
        fake = FakeRenderer()
        self.assertEqual(self.drive(fake=fake,max_captures=4),0)
        self.assertEqual(fake.calls,[(self.sid,b) for b in ('vis','mwir','lwir','ref_vis')])
        # The triplet is indexed as soon as it is complete; its references
        # join the manifest only once all three are published.
        self.assertEqual(self.index_flags(),[False])
        self.assertEqual(self.manifest(self.ref)['references'],{})
        self.assertIn('references=0/1',self.verify_output())
        more = FakeRenderer()
        self.assertEqual(self.drive(fake=more,max_captures=2),0)
        self.assertEqual(more.calls,[(self.sid,'ref_mwir'),(self.sid,'ref_lwir')])
        self.assertEqual(self.index_flags(),[True])


if __name__=='__main__':unittest.main()
