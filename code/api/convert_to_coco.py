import json
import argparse
import os 

parser = argparse.ArgumentParser()

parser.add_argument('-l', '--val_labels', type=str, required=False, help="val labels folder path")
parser.add_argument('-o', '--output_path', type=str, required=False, help="output json path")

args = parser.parse_args()

LABELS_PATH = "data/workspace/aug/val/labels/" if args.val_labels is None else args.val_labels
OUTPUT_PATH = "data/workspace/aug/annotations.json" if args.output_path is None else args.output_path

labels = sorted(os.listdir(LABELS_PATH), key=lambda x: int(x.split(".")[0]))

def pascalvoc_to_coco(coords):
    coords = [float(x) for x in coords]
    coords[2]-=coords[0]
    coords[3]-=coords[1]
    
    return coords


coco = {
        "info": {"year": "2023", "version": "1.0"},
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
            }
        ],
        "categories": [{"id": 1, "name": "face"}],
        "images": [],
        "annotations": [],
    }

index = 1
ann_index = 1
for label in labels:
    print("Writing record number " + str(index))
    with open(os.path.join(LABELS_PATH, label), "r") as f:
        annotations = json.load(f)

    image = {
        "id": index,
        "license": 1,
        "file_name": annotations["image"],
        "height": 640, 
        "width": 640, 
        "date_captured": None,
    }
    coco["images"].append(image)

    for j in range(len(annotations["bboxes"])):
        
        ann = {
            "id": ann_index,
            "image_id": index,
            "category_id": 1,
            "bbox": pascalvoc_to_coco(annotations["bboxes"][j]),
            "area": 1000, 
            "iscrowd": 0,
        }
        coco["annotations"].append(ann)
        ann_index += 1
        
    index += 1


with open(OUTPUT_PATH, "w") as f:
    json.dump(coco, f)
