# prevede vysledky yolo detektoru (.txt ve slozce labels) na result format

import json
import argparse
import os 
import cv2

parser = argparse.ArgumentParser()

parser.add_argument('-w', '--width', type=int, required=False, help="width of processing video (1280 default)")
parser.add_argument('-e', '--height', type=int, required=False, help="height of processing video (720 default)")
parser.add_argument('-l', '--labels', type=str, required=False, help="labels folder path")
parser.add_argument('-o', '--output_path', type=str, required=False, help="output txt file path")

args = parser.parse_args()

width = 1280 if args.width is None else args.width
height = 720 if args.height is None else args.height
LABELS_PATH = "labels" if args.labels is None else args.labels
OUTPUT_PATH = "output.txt" if args.output_path is None else args.output_path



def yolo_to_coco(coords, width, height):
    coords = [float(x) for x in coords]
    x1 = coords[0] - (coords[2]/2)
    y1 = coords[1] - (coords[3]/2)
    w = coords[2]
    h = coords[3]
    return x1 * width, y1 * height, w *width, h*height

labels = sorted(os.listdir(LABELS_PATH), key=lambda x: int(x.split(".")[0].split("_")[-1]))

output_file = open(OUTPUT_PATH, "w")

index = 1
for label in labels:
    print("Processing frame " + str(label))
    
      
    try:
        with open(os.path.join(LABELS_PATH, label), "r") as f:
            lines = f.readlines()
            
            output_file.write(f"Frame {index}\n")
            output_file.write(str(len(lines)) + "\n")
            
            for line in lines:
                bbox = yolo_to_coco(line.split()[1:5], width, height)
                bbox = [str(x) for x in bbox]
                output_file.write(" ".join(bbox))
                output_file.write("\n")
            
    except FileNotFoundError:
        output_file.write(f"Frame {index}\n")
        output_file.write("0\n")
    
    index +=1

output_file.close()
