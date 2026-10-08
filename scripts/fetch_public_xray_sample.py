"""How to fetch a small PUBLIC chest-X-ray sample for real-data experiments.

This script only prints instructions and (optionally) downloads a tiny
demonstration subset. Real radiographs are NEVER bundled with this project.

Recommended public datasets (details + licenses in docs/DATASETS.md):
  1. NIH ChestX-ray14 — 112k images, CC0-ish public release
     https://nihcc.app.box.com/v/ChestXray-NIHCC
  2. CheXpert (Stanford) — 224k studies, gated access for research
     https://stanfordmlgroup.github.io/competitions/chexpert/
  3. MIMIC-CXR (PhysioNet) — credentialed access required
     https://physionet.org/content/mimic-cxr/

Usage:
    python scripts/fetch_public_xray_sample.py --help
"""
import argparse
import os

TARGET = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "artifacts", "xray_sample")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=TARGET)
    ap.add_argument("--n", type=int, default=50,
                    help="number of sample images to fetch")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print("=" * 70)
    print("PUBLIC CHEST X-RAY SAMPLE — instructions (no auto-download by default)")
    print("=" * 70)
    print("""
1. NIH ChestX-ray14 (easiest, no login):
   - Open https://nihcc.app.box.com/v/ChestXray-NIHCC
   - Download one of the image archives (e.g. images_001.tar.gz, ~4 GB)
   - Extract and copy any N images into: %s

2. Then point the vision trainer at them:
     from models import vision as V
     model = V.get_vision_model(pretrained=True)
     model.fit(my_images, my_labels, epochs=3)

3. Labels: NIH provides Data_Entry_2017.csv with finding labels per image.
   Map the 14 finding columns to a binary 'abnormal' target, or train
   multi-label by replacing the classifier head (see models/vision.py).

IMPORTANT: real datasets carry their own licenses and intended-use terms
(see docs/DATASETS.md). Never mix real patient images into demos without
checking your institution's data-governance rules.
""" % args.out)
    print(f"Target folder ready: {args.out}")
    print("Done — nothing downloaded (manual step required).")


if __name__ == "__main__":
    main()
