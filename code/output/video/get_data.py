# get_data.py
# Vojtech Orava (xorava02)
# BP 2022/2023 FIT VUT
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--fps', type=str, required=False, action='append', nargs='*', help="FPS file from GUI app")
parser.add_argument('-d', '--detections', type=str, required=False, action='append', nargs='*', help="detection file from GUI app")

args = parser.parse_args()



def get_fps(fps_files):
    """Vytvori grafy z FPS souboru z GUI aplikace

    Args:
        fps_files (list): pole cest k FPS souborum
    """
    fig = plt.figure(figsize=(13,8))
    
    for i in fps_files[0]:
        x = []
        y = []
        with open(i, "r") as f:
            title = f.readline()
            plt.title(title.split("-")[0])
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                x.append(int(row[0]))
                y.append(float(row[1]))
                
        p = plt.plot(x,y, label=' '.join(title.split("-")[1:]).strip())
        plt.axhline(y=np.nanmean(y), linestyle='--', linewidth=3, label=f"Average FPS ({np.nanmean(y):.3f})", color=p[0].get_color())
        
    plt.ylabel("FPS")
    plt.xlabel("Frames")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()

def get_detections(detection_files):
    """Vytvori grafy z detekcnich souboru z GUI aplikace

    Args:
        detection_files (list): pole cest k detekcnim souborum
    """
    fig = plt.figure(figsize=(13,8))
    
    for i in detection_files[0]:
        x = []
        y = []
        with open(i, "r") as f:
            title = f.readline()
            plt.title(title.split("-")[0])
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                x.append(int(row[0]))
                y.append(float(row[1]))
                
        p = plt.plot(x,y, label=' '.join(title.split("-")[1:]).strip())
        plt.axhline(y=np.nanmean(y), linestyle='--', linewidth=3, label=f"Average detections ({np.nanmean(y):.3f})", color=p[0].get_color())
        
    plt.ylabel("Detections")
    plt.xlabel("Frames")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()

if args.fps != None:
    get_fps(args.fps)

if args.detections != None:
    get_detections(args.detections)