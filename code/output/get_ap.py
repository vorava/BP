# get_ap.py
# Vojtech Orava (xorava02)
# BP 2022/2023 FIT VUT
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-g', '--ground_truth', type=str, required=False, help="Ground truth file in widerface coco format from CVAT tool")
parser.add_argument('-d', '--detections', type=str, required=False, help="output detection file from GUI app")

args = parser.parse_args()


gt_path = 'gt.txt' if args.ground_truth is None else args.ground_truth
det_path = 'det.txt' if args.detections is None else args.detections

# vypocita IoU
# https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef
def compute_iou(boxes1, boxes2):
    """Vypocita IoU mezi bounding boxy
    Args:
        boxes1: (np.darray), tvar [N, 4] (x1, y1, x2, y2) pascal VOC
        boxes2: (np.darray), tvar [M, 4] (x1, y1, x2, y2) pascal VOC
    Returns:
        iou (np.array): IoU hodnota, tvar [N, M], N - ty GT box a M - ty detekovany box
    """
    # vypocitani plochy bboxu
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # vypocet pruniku
    x1 = np.maximum(boxes1[:, 0].reshape(-1, 1), boxes2[:, 0])
    x2 = np.minimum(boxes1[:, 2].reshape(-1, 1), boxes2[:, 2])
    y1 = np.maximum(boxes1[:, 1].reshape(-1, 1), boxes2[:, 1])
    y2 = np.minimum(boxes1[:, 3].reshape(-1, 1), boxes2[:, 3])
    
    inter_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    
    # vypocet sjednoceni
    union_area = area1.reshape(-1, 1) + area2 - inter_area
    
    # vypocet IoU
    iou = inter_area / np.maximum(union_area, np.finfo(np.float32).eps) # osetreni deleni nulou
    
    return iou


gtf = open(gt_path, "r")
detf = open(det_path, "r")

aps = []
ious_vals = []
# zpracovani obou souboru
while True:
    line = gtf.readline()
    if not line:
        break

    frame_name = line.rstrip()
    
    print(f"GT: {frame_name}")
    
    count = int(gtf.readline())
    gt_bboxes = np.empty((count, 4))
    if count == 0: 
        gtf.readline()
    else:
        for i in range(count):
            bbox = np.array(gtf.readline().split()[:4], dtype=np.float32)
            bbox[2] = bbox[2] + bbox[0]
            bbox[3] = bbox[3] + bbox[1]
            gt_bboxes[i] = (bbox.astype(int))
            
            
    line = detf.readline()
    if not line:
        break

    frame_name = line.rstrip()
    
    print(f"DET: {frame_name}")
    
    count = int(detf.readline())
    det_bboxes = np.empty((count, 4))
    if count == 0: 
        continue
    else:
        for i in range(count):
            bbox = np.array(detf.readline().split()[:4], dtype=np.float32)
            bbox[2] = bbox[2] + bbox[0]
            bbox[3] = bbox[3] + bbox[1]
            det_bboxes[i] = (bbox.astype(int))
            
    #print(gt_bboxes)
    #print(det_bboxes)
    
    # vypocita IoU 
    # v promenne ious [i,j] je IoU pro i-ty GT box a j-ty detekovany box
    ious = compute_iou(gt_bboxes, det_bboxes)
    # najde se kterym boxem ma IoU nejvyssi hodnotu
    detected_gt = np.argmax(ious, axis=0)
    
    ious_threshold = 0.5
    # pole presnosti a recallu
    precisions = np.zeros(len(det_bboxes))
    recalls = np.zeros(len(det_bboxes))
    
    # tru positive a false positive pole
    tp = np.zeros(len(det_bboxes))
    fp = np.zeros(len(det_bboxes))
    
    for i in range(len(det_bboxes)):
        #print(ious)
        # vybere se iou pro i-ty gt box a detekovany box s nejvyssi IoU
        iou = ious[detected_gt[i], i]
        ious_vals.append(iou)
        
        # vypocet dle https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/
        if iou >= ious_threshold:
            tp[i] = 1
        else:
            fp[i] = 1
            
        tpc = np.cumsum(tp)
        fpc = np.cumsum(fp)
        
        recalls[i] = tpc[-1] / float(len(gt_bboxes))
        precisions[i] = tpc[-1] / np.maximum(tpc[-1] + fpc[-1], np.finfo(np.float64).eps)    # ochrana deleni nulou
        
        #print(precisions[i])
        

    # osetreni vypoctu precision interpolaci (vezme se diky tomu jen nejvyssi cislo) - pro pripady, kdy jednomu x nalezi vice Y
    values, counts = np.unique(recalls, return_counts=True)
    old_precisions = precisions.copy()
    recalls = []
    for i in range(len(values)):
        if counts[i] > 1:
            new_vals = np.linspace(values[i], values[i]+0.0001, counts[i])  
            for j in range(len(new_vals)):
                recalls.append(new_vals[j])
        else:
            recalls.append(values[i])


    # 11 bodova interpolace
    recall_interp = np.linspace(0, 1, 11)
    #print(recall_interp)

    precision_interp = np.zeros_like(recall_interp)
    for i in range(len(precisions)):
        precisions[i] = np.max(precisions[i:])

    precision_interp = (np.interp(recall_interp, recalls, precisions, right=0))
    #print("----------------")
    #print(precision_interp)

    ap = (1/11) * np.sum(precision_interp)
    aps.append(ap)
    """
    print(precisions)
    print(recalls)
    
    fig = plt.figure(figsize=(13,8))
    plt.plot(recalls, precisions)
    plt.plot(recalls, old_precisions)
    plt.ylim(0,1.05)
    plt.xlim(0,1.05)
    plt.show()"""
    
    print(f"Current AP: {ap}")
    
print("Final AP: ", end="")
print(np.sum(aps)/len(aps))

print("Final IoU: ", end="")
print(np.sum(ious_vals)/len(ious_vals))
