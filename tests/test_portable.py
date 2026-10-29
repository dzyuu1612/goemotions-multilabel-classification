import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from src.models.portable import unpack_model


class PortableTest(unittest.TestCase):
    def bundle(self,path,name='checkpoint/model.safetensors',payload=b'ok',expected=None):
        m={'format_version':1,'label_names':[str(i) for i in range(28)],'thresholds':[.5]*28,
           'files_sha256':{name:expected or hashlib.sha256(payload).hexdigest()}}
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('manifest.json',json.dumps(m));z.writestr(name,payload)

    def test_extract_repairs_changed_cache_and_rejects_bad_hash(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'model.goemotions';self.bundle(p)
            root,_=unpack_model(p,d/'cache')
            (root/'checkpoint/model.safetensors').write_bytes(b'changed')
            unpack_model(p,d/'cache')
            self.assertEqual((root/'checkpoint/model.safetensors').read_bytes(),b'ok')
            self.bundle(p,expected='0'*64)
            with self.assertRaisesRegex(ValueError,'checksum'):unpack_model(p,d/'cache')

    def test_member_cannot_escape_cache(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'model.goemotions';self.bundle(p,'checkpoint/../../escaped')
            with self.assertRaisesRegex(ValueError,'path'):unpack_model(p,d/'cache')
            self.assertFalse((d/'escaped').exists())
