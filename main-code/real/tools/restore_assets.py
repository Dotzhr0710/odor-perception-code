"""Restore byte-identical data assets from platform-sized parts (stdlib only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(relative):
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError("Asset path is outside the release")
    return path


def restore(verify_only=False):
    manifest = json.loads((ROOT / "ASSETS.json").read_text(encoding="utf-8"))
    for entry in manifest["assets"]:
        target = safe_path(entry["path"])
        parts = entry.get("parts", [])
        if parts:
            for part in parts:
                path = safe_path(part["path"])
                if not path.is_file() or path.stat().st_size != part["size"] or sha(path) != part["sha256"]:
                    raise RuntimeError("Missing or altered data part: " + part["path"])
        if target.exists():
            if target.stat().st_size != entry["size"] or sha(target) != entry["sha256"]:
                raise RuntimeError("Existing asset differs; preserved without overwrite: " + entry["path"])
        elif parts:
            if verify_only:
                h = hashlib.sha256()
                size = 0
                for part in parts:
                    with safe_path(part["path"]).open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            h.update(block)
                            size += len(block)
                if size != entry["size"] or h.hexdigest() != entry["sha256"]:
                    raise RuntimeError("Assembled asset checksum mismatch")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".assembling-" + str(os.getpid()))
                with temporary.open("xb") as output:
                    for part in parts:
                        with safe_path(part["path"]).open("rb") as handle:
                            for block in iter(lambda: handle.read(1024 * 1024), b""):
                                output.write(block)
                if temporary.stat().st_size != entry["size"] or sha(temporary) != entry["sha256"]:
                    raise RuntimeError("Restored asset failed checksum; original parts preserved")
                os.replace(str(temporary), str(target))
        else:
            raise FileNotFoundError("Missing asset: " + entry["path"])
    print("Assets verified" if verify_only else "Assets restored and verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Check parts without creating large files")
    restore(parser.parse_args().verify)
