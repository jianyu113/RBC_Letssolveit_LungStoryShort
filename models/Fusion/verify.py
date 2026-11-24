"""Verify the adopted checkpoints and optional unchanged prepared metadata."""

import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((REPO / "web/models.json").read_text())
    inputs = {
        "detector": (args.detector, manifest["detector"]["sha256"]),
        "classifier": (args.classifier, manifest["classifier"]["sha256"]),
    }
    if args.prepared_dir is not None:
        inputs.update({
            "labels.csv": (args.prepared_dir / "labels.csv", manifest["prepared_labels_sha256"]),
            "boxes.csv": (args.prepared_dir / "boxes.csv", manifest["prepared_boxes_sha256"]),
        })
    checked = {}
    for name, (path, expected) in inputs.items():
        if not path.is_file():
            parser.error(f"Missing {name}: {path}")
        actual = sha256(path)
        if actual != expected:
            parser.error(f"Wrong or incomplete {name}: SHA256 {actual}, expected {expected}")
        checked[name] = {"path": str(path.resolve()), "sha256": actual}
    print(json.dumps({
        "status": "PASS", "checked": checked, "fusion": manifest["fusion"],
        "scope": "Complete checkpoint bytes and optional metadata; PNG pixels and inference are not checked.",
    }, indent=2))


if __name__ == "__main__":
    main()
