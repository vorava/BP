import json
import argparse
import os 
import cv2

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--images', type=str, required=False, help="path to folder with images")
parser.add_argument('-l', '--labels', type=str, required=False, help="labels folder path")
parser.add_argument('-o', '--output_path', type=str, required=False, help="output json path")

args = parser.parse_args()

IMAGES_PATH = "../api/data/workspace/aug/val/images" if args.images is None else args.images
LABELS_PATH = "labels" if args.labels is None else args.labels
OUTPUT_PATH = "results.json" if args.output_path is None else args.output_path

images = sorted(os.listdir(IMAGES_PATH), key=lambda x: int(x.split(".")[0]))

results = []

def yolo_to_coco(coords, width, height):
    coords = [float(x) for x in coords]
    x1 = coords[0] - (coords[2]/2)
    y1 = coords[1] - (coords[3]/2)
    w = coords[2]
    h = coords[3]
    return x1 * width, y1 * height, w *width, h*height


index = 1
ann_index = 1
for img in images:
    print("Processing file " + str(img))
    
    # nacteni sirky a vysky obrazku
    filename = img.split(".jpg")[0] + ".txt"
    
    image = cv2.imread(os.path.join(IMAGES_PATH, img))
    height, width, _ = image.shape
    
    try:
        with open(os.path.join(LABELS_PATH, filename), "r") as f:
            lines = f.readlines()
            
            for line in lines:
                bbox = yolo_to_coco(line.split()[1:5], width, height)
                data = {
                    "image_id": index,
                    "category_id": 1,
                    "bbox": bbox,
                    "score": 1.0, 
                }
                results.append(data)
            
    except FileNotFoundError:
        pass
    
    index +=1

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f)
