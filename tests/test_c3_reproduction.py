"""Integrity guards must reject changed or non-full C3 training artifacts."""
import tempfile
from pathlib import Path
import unittest

from scripts.distilbert.reproduce_c3 import checked_run
from src.datasets.goemotions import sha256
from src.models.distilbert_study import write_json


class C3ReproductionTest(unittest.TestCase):
    def test_resume_rejects_changed_checkpoint_before_reusing_a_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'weights').write_bytes(b'original checkpoint')
            config={'seed':42}
            meta={'status':'complete','mode':'full','config':config,'source_sha256':{},
                  'n_train':43410,'n_validation':5426,'test_used':False,
                  'artifact_sha256':{'weights':sha256(root/'weights')}}
            write_json(root/'run.json',meta)
            self.assertEqual(checked_run(root,config,{}),meta)
            (root/'weights').write_bytes(b'other checkpoint')
            with self.assertRaisesRegex(ValueError,'Changed training artifact'):
                checked_run(root,config,{})

    def test_pilot_test_contamination_or_config_drift_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config={'seed':42}
            meta={'status':'complete','mode':'full','config':config,'source_sha256':{},
                  'n_train':43410,'n_validation':5426,'test_used':False,'artifact_sha256':{}}
            for change in ({'mode':'pilot'},{'n_train':100},{'test_used':True},
                           {'config':{'seed':123}},{'source_sha256':{'trainer':'changed'}}):
                write_json(root/'run.json',dict(meta,**change))
                with self.subTest(change=change),self.assertRaises(ValueError):
                    checked_run(root,config,{})


if __name__=='__main__':
    unittest.main()
