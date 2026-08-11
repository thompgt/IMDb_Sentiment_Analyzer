"""Fetch the IMDb review corpus and write ``data/imdb_reviews.csv``.

This is the **only** supported data pipeline. It downloads the Stanford
aclImdb v1 tarball, which needs no credentials -- unlike the kagglehub call the
notebook used to make, which silently required a Kaggle API token that nothing
in the repo told you to configure.

Everything it writes goes under ``data/`` (git-ignored). The previous version
dropped a ~66 MB CSV and an 84 MB tarball straight into the repo root, one
``git add .`` away from being committed.

Usage::

    python download_data.py                 # -> data/imdb_reviews.csv
    python download_data.py --force         # re-extract and rebuild
    python download_data.py --keep-archive  # don't delete the tarball
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

#: Published checksum for aclImdb_v1.tar.gz. A truncated or tampered download
#: is caught here rather than surfacing as a confusing tarfile error later.
EXPECTED_SHA256 = "c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe"

DATA_DIR = Path("data")
ARCHIVE_PATH = DATA_DIR / "aclImdb_v1.tar.gz"
EXTRACT_ROOT = DATA_DIR / "aclImdb_extracted"
OUTPUT_CSV = DATA_DIR / "imdb_reviews.csv"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest``, verifying the SHA-256 before returning."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and sha256_of(dest) == EXPECTED_SHA256:
        print(f"Archive already present and verified: {dest}")
        return dest

    print(f"Downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with tmp.open("wb") as fh, tqdm(
            total=total, unit="iB", unit_scale=True, desc="download"
        ) as bar:
            for block in response.iter_content(1024 * 64):
                fh.write(block)
                bar.update(len(block))

    actual = sha256_of(tmp)
    if actual != EXPECTED_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {url}\n  expected {EXPECTED_SHA256}\n  got      {actual}"
        )
    tmp.replace(dest)
    print("Download complete and checksum verified.")
    return dest


def extract(archive: Path, root: Path, *, force: bool = False) -> Path:
    """Safely extract the tarball.

    ``filter="data"`` refuses absolute paths, ``..`` traversal, symlinks and
    device files. The original code called bare ``extractall()``, which happily
    writes anywhere the archive tells it to -- and which Python 3.14 rejects
    outright.
    """
    marker = root / "aclImdb"
    if marker.is_dir() and not force:
        print(f"Already extracted: {marker}")
        return marker
    if force and root.exists():
        shutil.rmtree(root)

    root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} -> {root}")
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(path=root, filter="data")
        except TypeError:  # Python < 3.12 has no `filter` argument
            tar.extractall(path=root)  # noqa: S202
    print("Extraction complete.")
    return marker


def build_dataframe(corpus_root: Path) -> pd.DataFrame:
    """Read every labelled review from the train and test splits."""
    rows: list[dict[str, str]] = []
    for split in ("train", "test"):
        for label in ("pos", "neg"):
            directory = corpus_root / split / label
            if not directory.is_dir():
                raise FileNotFoundError(f"expected corpus directory {directory}")
            files = sorted(directory.glob("*.txt"))
            if not files:
                raise FileNotFoundError(
                    f"{directory} contains no .txt reviews - the extraction is "
                    f"incomplete. Re-run with --force."
                )
            for path in tqdm(files, desc=f"{split}/{label}", unit="file"):
                rows.append(
                    {
                        "review": path.read_text(encoding="utf-8"),
                        "sentiment": "positive" if label == "pos" else "negative",
                        "split": split,
                    }
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="re-extract and rebuild even if outputs exist")
    parser.add_argument("--keep-archive", action="store_true",
                        help="keep the 84 MB tarball after building the CSV")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        print(f"{args.output} already exists (use --force to rebuild).")
        return 0

    try:
        archive = download(DATA_URL, ARCHIVE_PATH)
        corpus_root = extract(archive, EXTRACT_ROOT, force=args.force)
        df = build_dataframe(corpus_root)
    except (requests.RequestException, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} reviews to {args.output}")

    if not args.keep_archive:
        archive.unlink(missing_ok=True)
        print(f"Removed {archive} (pass --keep-archive to retain it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
