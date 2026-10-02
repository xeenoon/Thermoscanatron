"""Build a list of hand-free images from COCO to use as "no hand" training data and as paste backgrounds.

segkit-negatives data/coco --split val2017 --out data/negatives_val.txt

Keeps images with no `person` annotation (COCO labels people, including crowds and partial bodies),
then double-checks with MediaPipe and drops any image where it still finds a hand.
"""

import argparse
import json
from pathlib import Path

import cv2

from segkit.hand_landmarks import HandLandmarks


def main() -> None:
    p = argparse.ArgumentParser(prog="segkit-negatives")
    p.add_argument("coco", type=Path, help="dir containing annotations/ and <split>/")
    p.add_argument("--split", default="val2017")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-hand-check", action="store_true", help="skip the MediaPipe pass")
    args = p.parse_args()

    ann = json.loads((args.coco / "annotations" / f"instances_{args.split}.json").read_text())
    person_id = next(c["id"] for c in ann["categories"] if c["name"] == "person")
    with_person = {a["image_id"] for a in ann["annotations"] if a["category_id"] == person_id}
    candidates = [args.coco / args.split / im["file_name"] for im in ann["images"] if im["id"] not in with_person]
    print(f"{args.split}: {len(ann['images'])} images, {len(candidates)} without a person annotation")

    keep = candidates
    if not args.no_hand_check:
        detector = HandLandmarks()
        keep = []
        for i, path in enumerate(candidates, 1):
            bgr = cv2.imread(str(path))
            if bgr is not None and not detector.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)):
                keep.append(path)
            if i % 2000 == 0:
                print(f"  hand check {i}/{len(candidates)}: kept {len(keep)}", flush=True)
        detector.close()
        print(f"MediaPipe found a hand in {len(candidates) - len(keep)} of them; dropped")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(str(p.resolve()) for p in keep) + "\n")
    print(f"{len(keep)} negative images -> {args.out}")


if __name__ == "__main__":
    main()
