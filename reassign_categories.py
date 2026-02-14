"""
Author: Pavel Timonin
Created: 2026-02-08
Description: Reassigns category IDs in a COCO dataset to be contiguous.
"""

from pycocotools.coco import COCO


def reassign_category_ids(coco: COCO) -> None:
    """
    Reassigns category IDs to be contiguous starting from 1.

    Modifies the COCO object in-place and rebuilds the index.

    Args:
        coco (COCO): The COCO object containing the dataset.
    """
    # Get all categories
    categories = coco.dataset.get("categories", [])
    if not categories:
        return

    # Sort categories by ID
    categories.sort(key=lambda x: x["id"])

    # Create mapping
    old_to_new = {}
    new_categories = []
    for new_id, cat in enumerate(categories, start=1):
        old_id = cat["id"]
        old_to_new[old_id] = new_id
        cat["id"] = new_id
        new_categories.append(cat)

    coco.dataset["categories"] = new_categories

    # Update annotations
    annotations = coco.dataset.get("annotations", [])
    for ann in annotations:
        # Check for Panoptic format
        if "segments_info" in ann:
            for seg in ann["segments_info"]:
                old_cat_id = seg["category_id"]
                if old_cat_id in old_to_new:
                    seg["category_id"] = old_to_new[old_cat_id]
        # Check for Instance format
        elif "category_id" in ann:
            old_cat_id = ann["category_id"]
            if old_cat_id in old_to_new:
                ann["category_id"] = old_to_new[old_cat_id]

    # Rebuild index
    coco.createIndex()

    print(
        f"Reassigned {len(old_to_new)} categories to contiguous IDs 1..{len(old_to_new)}."
    )
