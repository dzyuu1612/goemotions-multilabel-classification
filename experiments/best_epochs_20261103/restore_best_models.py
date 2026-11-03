"""Download one checked model package; no training occurs."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    manifest=json.loads((ROOT/"model_package.json").read_text(encoding="utf-8"))
    archive=ROOT/manifest["filename"]
    if not archive.exists() or sha(archive)!=manifest["sha256"]:
        temp=archive.with_suffix(".download")
        request=urllib.request.Request(manifest["download_url"],headers={"User-Agent":"GoEmotions-best-epochs"})
        with urllib.request.urlopen(request,timeout=120) as source,temp.open("wb") as output:
            while block:=source.read(8*1024*1024):output.write(block)
        if sha(temp)!=manifest["sha256"]:raise ValueError("Downloaded model package has a different SHA-256")
        temp.replace(archive)
    destination=(ROOT/"checkpoints").resolve()
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        files=[i for i in bundle.infolist() if not i.is_dir()]
        if {i.filename for i in files}!=set(manifest["files_sha256"]):raise ValueError("Model ZIP contains unexpected files")
        for item in files:
            target=(destination/item.filename).resolve()
            if not target.is_relative_to(destination):raise ValueError("Unsafe archive path")
            if target.exists():
                if sha(target)!=manifest["files_sha256"][item.filename]:raise ValueError("Existing checkpoint differs: "+item.filename)
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with bundle.open(item) as source,target.open("wb") as output:
                while block:=source.read(8*1024*1024):output.write(block)
            if sha(target)!=manifest["files_sha256"][item.filename]:raise ValueError("Checkpoint extraction hash mismatch")
    print("Restored three representative C checkpoints; no training.")


if __name__=="__main__":main()
