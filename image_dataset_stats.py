import argparse
import json
import math
from collections import Counter
from pathlib import Path

def dataset_stats(dataset_dir: Path):
    with (dataset_dir / "annotations.json").open("r", encoding="utf-8") as f:
        records = json.load(f)

    labels_by_image = {}
    classes = set()

    for record in records:
        for ann in record["annotations"]:
            image_path = ann["image_path"]
            class_label = ann["class_label"]

            classes.add(class_label)
            if image_path not in labels_by_image:
                labels_by_image[image_path] = Counter()
            labels_by_image[image_path][class_label] += 1

    entropies = []
    for counts in labels_by_image.values():
        labels = []
        total_count = sum([count for _, count in enumerate(counts.values())])
        entropy = 0
        for class_id, count in enumerate(counts.values()):
            class_prob = count/total_count
            entropy += -class_prob*math.log(class_prob)
        entropies.append(entropy)

    dataset_size = len(labels_by_image)
    num_classes = len(classes)
    avg_entropy = sum(entropies) / len(entropies)
    return dataset_size, num_classes, avg_entropy


def main(im_path = "data/image"):
    dataset_dirs = sorted([p for p in Path(im_path).iterdir() if p.is_dir()])

    for dataset_dir in dataset_dirs:
        dataset_size, n_classes, avg_entropy = dataset_stats(dataset_dir)
        print(f"{dataset_dir.name}")
        print(f"  dataset_size: {dataset_size}")
        print(f"  num_classes: {n_classes}")
        print(f"  avg_entropy: {avg_entropy:.6f}")


if __name__ == "__main__":
    main()