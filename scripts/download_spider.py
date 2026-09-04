# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Fetch the Spider Text-to-SQL benchmark and lay it out for the Dr. Zero
# Text-to-SQL adaptation.
#
# Target layout (consumed by search/sql_exec_server.py and process_*.py):
#   <save_path>/database/<db_id>/<db_id>.sqlite   -> SQLite databases (needed to execute!)
#   <save_path>/tables.json                       -> schema descriptions
#   <save_path>/train_spider.json                 -> gold NL/SQL pairs (train)
#   <save_path>/train_others.json                 -> extra gold pairs (train)
#   <save_path>/dev.json                          -> gold NL/SQL pairs (eval)
#
# IMPORTANT: the HF dataset `xlangai/spider` ships ONLY parquet NL/SQL pairs, NOT
# the .sqlite databases. Execution-based reward requires the databases, which come
# only from the official Spider zip (`spider_data.zip` from
# https://yale-lily.github.io/spider). This script therefore supports three sources
# and a single normalization step. Pick whichever is convenient:
#
#   # (a) already-downloaded zip (most reliable):
#   python scripts/download_spider.py --zip_path /path/to/spider_data.zip
#
#   # (b) Google Drive id (needs `pip install gdown`):
#   python scripts/download_spider.py --gdrive_id <FILE_ID>
#
#   # (c) a HF dataset mirror that DOES include a database/ folder:
#   python scripts/download_spider.py --repo_id <org/repo>

import argparse
import os
import shutil
import zipfile


GOLD_FILES = ["tables.json", "train_spider.json", "train_others.json", "dev.json"]


# ----------------------------------------------------------------------------
# Sources -> a raw directory to normalize
# ----------------------------------------------------------------------------

def _from_zip(zip_path: str, work_dir: str) -> str:
    if not os.path.exists(zip_path):
        raise SystemExit(f"[spider] zip not found: {zip_path}")
    print(f"[spider] extracting {zip_path} -> {work_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work_dir)
    return work_dir


def _from_gdrive(gdrive_id: str, work_dir: str) -> str:
    try:
        import gdown
    except ImportError:
        raise SystemExit("[spider] --gdrive_id needs `pip install gdown`.")
    dst = os.path.join(work_dir, "spider_data.zip")
    print(f"[spider] downloading Google Drive id {gdrive_id} -> {dst}")
    gdown.download(id=gdrive_id, output=dst, quiet=False)
    return _from_zip(dst, work_dir)


def _from_hf(repo_id: str, work_dir: str) -> str:
    from huggingface_hub import snapshot_download

    print(f"[spider] downloading snapshot of {repo_id} (dataset)...")
    path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=work_dir)
    _unzip_all(path)  # some mirrors nest a zip
    return path


# ----------------------------------------------------------------------------
# Normalization helpers
# ----------------------------------------------------------------------------

def _unzip_all(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".zip"):
                zpath = os.path.join(dirpath, fn)
                print(f"[spider] unzipping nested {zpath}")
                try:
                    with zipfile.ZipFile(zpath) as zf:
                        zf.extractall(dirpath)
                except zipfile.BadZipFile:
                    print(f"[spider]   skip (not a valid zip): {zpath}")


def _find_db_dir(root: str):
    """First directory named 'database' that actually contains .sqlite files."""
    for dirpath, _, _ in os.walk(root):
        if os.path.basename(dirpath) == "database":
            if any(f.endswith(".sqlite")
                   for _, _, fs in os.walk(dirpath) for f in fs):
                return dirpath
    return None


def _find_file(root: str, name: str):
    for dirpath, _, filenames in os.walk(root):
        if name in filenames:
            return os.path.join(dirpath, name)
    return None


def _normalize(raw_root: str, save_path: str):
    os.makedirs(save_path, exist_ok=True)

    db_src = _find_db_dir(raw_root)
    if db_src is None:
        raise SystemExit(
            "[spider] no 'database/' folder with .sqlite files found.\n"
            "         The parquet-only HF mirror (xlangai/spider) does NOT include\n"
            "         databases. Download spider_data.zip from\n"
            "         https://yale-lily.github.io/spider and pass it via --zip_path."
        )
    db_dst = os.path.join(save_path, "database")
    if os.path.abspath(db_src) != os.path.abspath(db_dst):
        if os.path.exists(db_dst):
            print(f"[spider] removing existing {db_dst}")
            shutil.rmtree(db_dst)
        print(f"[spider] moving databases {db_src} -> {db_dst}")
        shutil.move(db_src, db_dst)

    n_db = sum(
        1 for d in os.listdir(db_dst)
        if os.path.isfile(os.path.join(db_dst, d, f"{d}.sqlite"))
    )
    print(f"[spider] {n_db} databases available under {db_dst}")

    for name in GOLD_FILES:
        src = _find_file(raw_root, name)
        dst = os.path.join(save_path, name)
        if src is None:
            print(f"[spider] WARNING: {name} not found (skipping)")
            continue
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy(src, dst)
        print(f"[spider] {name} -> {dst}")

    print("[spider] done.")


def main():
    parser = argparse.ArgumentParser(description="Download/lay out the Spider benchmark.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip_path", type=str, help="Path to an already-downloaded spider_data.zip.")
    src.add_argument("--gdrive_id", type=str, help="Google Drive file id of spider_data.zip.")
    src.add_argument("--repo_id", type=str, help="HF dataset repo that includes a database/ folder.")
    parser.add_argument("--save_path", type=str, default="./spider",
                        help="Destination root (gets database/ + gold JSONs).")
    parser.add_argument("--work_dir", type=str, default="./spider/_raw",
                        help="Scratch dir for the raw download before normalizing.")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    if args.zip_path:
        raw = _from_zip(args.zip_path, args.work_dir)
    elif args.gdrive_id:
        raw = _from_gdrive(args.gdrive_id, args.work_dir)
    else:
        raw = _from_hf(args.repo_id, args.work_dir)

    _normalize(raw, args.save_path)


if __name__ == "__main__":
    main()
