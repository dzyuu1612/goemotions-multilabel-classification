"""Integrity checks for the frozen handoff, without model/network downloads."""
from pathlib import Path
import hashlib,json,tempfile,unittest,zipfile
from unittest.mock import patch
import src.final_submission as final

class RuntimeIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.published=self.root/'published';self.published.mkdir()
        self.payload=b'official score receipt\n';self.name='data/labels.json'
        with zipfile.ZipFile(self.published/'runtime_bundle.zip','w') as z:z.writestr(self.name,self.payload)
        self.manifest={'runtime_bundle_sha256':final.digest(self.published/'runtime_bundle.zip'),'files_sha256':{self.name:hashlib.sha256(self.payload).hexdigest()}}
        self.save();self.mock=patch.object(final,'PUBLISHED',self.published);self.mock.start();final.prepare_runtime.cache_clear()
    def save(self):
        (self.published/'manifest.json').write_text(json.dumps(self.manifest),encoding='utf8')
    def tearDown(self):
        final.prepare_runtime.cache_clear();self.mock.stop();self.tmp.cleanup()
    def test_prepare_restores_exact_bytes_without_training(self):
        destination=final.prepare_runtime(self.root/'runtime')
        self.assertEqual((destination/self.name).read_bytes(),self.payload)
    def test_existing_different_artifact_is_not_overwritten(self):
        destination=self.root/'runtime';(destination/'data').mkdir(parents=True)
        (destination/self.name).write_bytes(b'another run')
        with self.assertRaisesRegex(ValueError,'hiện có khác'):final.prepare_runtime(destination)
        self.assertEqual((destination/self.name).read_bytes(),b'another run')
    def test_tampered_archive_fails_before_restoring(self):
        (self.published/'runtime_bundle.zip').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'khác manifest'):final.prepare_runtime(self.root/'runtime')
        self.assertFalse((self.root/'runtime').exists())
    def test_extra_zip_members_are_rejected(self):
        with zipfile.ZipFile(self.published/'runtime_bundle.zip','a') as z:z.writestr('unlisted',b'x')
        self.manifest['runtime_bundle_sha256']=final.digest(self.published/'runtime_bundle.zip');self.save()
        with self.assertRaisesRegex(ValueError,'khác manifest'):final.prepare_runtime(self.root/'runtime')
    def test_zip_path_cannot_escape_destination(self):
        self.name='../outside.txt';self.manifest['files_sha256']={self.name:hashlib.sha256(self.payload).hexdigest()}
        with zipfile.ZipFile(self.published/'runtime_bundle.zip','w') as z:z.writestr(self.name,self.payload)
        self.manifest['runtime_bundle_sha256']=final.digest(self.published/'runtime_bundle.zip');self.save()
        with self.assertRaisesRegex(ValueError,'vượt thư mục'):final.prepare_runtime(self.root/'runtime')
        self.assertFalse((self.root/'outside.txt').exists())

if __name__=='__main__':unittest.main()
