"""
Download the MISMIP+ ensemble results (Cornford et al., 2020, supplement: Zenodo
doi:10.5281/zenodo.3611936, supplement.zip, 20.7 MB) to raw/ensemble/ and unzip it
(raw/ensemble/submission_data/<Exp>_<submission>.nc, 250 files; raw/ is not versioned).

Usage:  python prep/fetch_ensemble.py
"""

import hashlib
import os
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

URL = "https://zenodo.org/records/3611936/files/supplement.zip?download=1"
MD5 = "d2704a2a5cca5433a09d7858ed78336d"
DEST = os.path.join(C.ROOT, "raw", "ensemble")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(DEST, exist_ok=True)
    zpath = os.path.join(DEST, "supplement.zip")
    if not (os.path.exists(zpath) and md5(zpath) == MD5):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zpath)
    if md5(zpath) != MD5:
        raise SystemExit(f"md5 mismatch for {zpath}")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DEST)
    n = len(os.listdir(os.path.join(DEST, "submission_data")))
    print(f"{zpath}: md5 ok, {n} submission files in {os.path.relpath(DEST, C.ROOT)}/submission_data")


if __name__ == "__main__":
    main()
