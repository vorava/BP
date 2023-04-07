from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument('-r', '--results', type=str, required=False, help="path to folder with results")
parser.add_argument('-o', '--output_path', type=str, required=False, help="output file path")
parser.add_argument('-a', '--annotation', type=str, required=False, help="annotation file path")

args = parser.parse_args()

RESULTS_PATH = "results" if args.results is None else args.results
OUTPUT_PATH = "precision_recall.txt" if args.output_path is None else args.output_path
ANNOTATION_PATH = "/mnt/c/VUTFIT/BP/code/api/data/workspace/aug/annotations.json" if args.annotation is None else args.annotation

def get_results(path):
    files = []
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            f_path = os.path.join(dirpath, f)
            files.append(f_path)

    return files


annotation_type = "bbox"

files = get_results(RESULTS_PATH)

annotation_file = ANNOTATION_PATH
coco_gt = COCO(annotation_file)

with open(OUTPUT_PATH, "w") as output_file:
    
    for f in files:
        print(f"Evaluating file {f}")
        
        output_file.write(f + "\n")
        
        results_file = f
        coco_dt = coco_gt.loadRes(results_file)
        
        images_ids = sorted(coco_gt.getImgIds())        

        cocoEval = COCOeval(coco_gt, coco_dt, annotation_type)
        cocoEval.params.imgIds  = images_ids
        cocoEval.evaluate()
        cocoEval.accumulate()

        sys.stdout = output_file
        cocoEval.summarize()
        sys.stdout = sys.__stdout__
        output_file.write("\n\n")