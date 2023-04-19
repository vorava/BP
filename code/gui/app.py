# app.py
# Vojtech Orava (xorava02), vedouci Ing. Tomas Goldmann
# BP 2022/2023, FIT VUT Brno

# importy pro GUI
from PyQt6.QtWidgets import *
import sys
from PyQt6.QtCore import *
import time
import cv2
import os
import numpy as np
from PyQt6.QtGui import *
from pathlib import Path
from functools import partial
import csv
import datetime
from mtcnn import MTCNN

# tf importy
import tensorflow as tf

# importy pro detekci
from openvino.runtime import Core

# krok preskoceni ve videu 
SKIP_VAL = 5 # sekundy

# OpenVINO vychozi nastaveni
ie = Core()

# promenne openvino modelu a TF modelu
global model
global compiled_model

model_tf_id = 0

# VJ
face_cascade = cv2.CascadeClassifier()
face_cascade.load("models/Viola-Jones/haarcascade_frontal.xml")

# MTCNN
mtcnn_detector = MTCNN()
    
@tf.function
def detect_faces_tf(model, image):
    """Detekuje obliceje v obrazku

    Args:
        image (np.darray): numpy reprezentace obrazku pro detekci 

    Returns:
        bboxy obliceju (pascalvoc), confidence detekce
    """
    
    
    input_tensor = tf.convert_to_tensor(image)
    
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    
    return detections['detection_boxes'][0], detections["detection_scores"][0]


def get_models():
    """Nacte modely, ktere je mozno pouzit k detekci (modely ze slozky models)

    Returns:
        list: pole modelu jako string
    """
    models = sorted(os.listdir("models"))
    return models


def get_devices():
    """Nacte pouzitelne zarizeni Intel (VPU, CPU, GPU)

    Returns:
        list: pole zarizeni jako dictionary {full_name, type}
    """
    device_list = []
    try:
        device_list.append({"full_name": ie.get_property("MYRIAD", "FULL_DEVICE_NAME"),
                            "type": "MYRIAD"})
    except Exception:
        pass
    
    try:
        device_list.append({"full_name": ie.get_property("CPU", "FULL_DEVICE_NAME"),
                            "type": "CPU"})
    except Exception:
        pass
    
    try:
        device_list.append({"full_name": ie.get_property("GPU", "FULL_DEVICE_NAME"),
                            "type": "GPU"})
    except Exception:
        pass
    
    return device_list

# pomocne detekcni funkce
def apply_bboxes(image, bboxes, conf, threshold=0.5, color = (0,0,255)):
    """ Prida do framu bounding boxy 

    Args:
        image (np.darray): zpracovavany frame ve vychozim rozliseni
        bboxes (np.darray): bboxy z funkce detect_faces(_tf)
        conf (np.darray): pole confidence pro jednotlive bboxy z funkce detect_faces(_tf)
        threshold (prah conf pro zobrazeni, optional): prahova hodnota verohodnosti. Defaults to 0.5.
        color (tuple, optional): barva bboxu

    Returns:
        image: puvodni obrazek jako np array s pridanymi bboxy
        nof_detections: pocet detekovanych obliceju, 
        output_bboxes: souradnice ve formatu coco (xywh) pro vystup do txt
    """
    DEFAULT_RESOLUTION = [640, 640] # rozliseni ssd detektoru
    bboxes[:, ::2] *= DEFAULT_RESOLUTION[0] # transformace bboxu z 0..1 na 0..640
    bboxes[:, 1::2] *= DEFAULT_RESOLUTION[1]
    
    # zjisteni pomeru stran vychoziho obrazku a vystupu ssd detektoru
    ratio_x, ratio_y = image.shape[1] / DEFAULT_RESOLUTION[1], image.shape[0] / DEFAULT_RESOLUTION[0]
    
    nof_detections = 0
    output_bboxes = []
    # iterace nad bboxy a zakresleni obdelniku a conf
    for i, box in enumerate(bboxes):
    
        # tato cast vychazi z https://hub.gke2.mybinder.org/user/openvinotoolkit-nvino_notebooks-59sp5s7q/notebooks/notebooks/004-hello-detection/004-hello-detection.ipynb
        if(conf[i] > threshold):
            nof_detections+=1
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y, 10)) if idx % 2 
                else int(corner_position * ratio_x)
                for idx, corner_position in enumerate(box)
            ]           
            
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                      
            image = cv2.putText(
                    image,
                    f"{float(conf[i]):.2f}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,0,0),
                    2,
                    cv2.LINE_AA,
                )
            
            
            # konec castecne prevzate casti
            output_bboxes.append([x_min, y_min, x_max - x_min, y_max-y_min])
            
    return image, nof_detections, output_bboxes

def apply_bboxes_tf(image, bboxes, conf, threshold=0.5, color = (0,0,255)):
    """ Prida do framu bounding boxy 

    Args:
        image (np.darray): zpracovavany frame ve vychozim rozliseni
        bboxes (np.darray): bboxy z funkce detect_faces(_tf)
        conf (np.darray): pole confidence pro jednotlive bboxy z funkce detect_faces(_tf)
        threshold (prah conf pro zobrazeni, optional): prahova hodnota verohodnosti. Defaults to 0.5.
        color (tuple, optional): barva bboxu

    Returns:
        image: puvodni obrazek jako np array s pridanymi bboxy
        nof_detections: pocet detekovanych obliceju, 
        output_bboxes: souradnice ve formatu coco (xywh) pro vystup do txt
    """
    bboxes[:,1::2] *= image.shape[1] # width
    bboxes[:,::2] *= image.shape[0] #height
    
    nof_detections = 0
    output_bboxes = []
    # iterace nad bboxy a zakresleni obdelniku a conf
    for i, box in enumerate(bboxes):
        
        # tato cast vychazi z https://hub.gke2.mybinder.org/user/openvinotoolkit-nvino_notebooks-59sp5s7q/notebooks/notebooks/004-hello-detection/004-hello-detection.ipynb
        if(conf[i] > threshold):
            nof_detections+=1
            box = [int(x) for x in box]
            (x_min, y_min, x_max, y_max) = box[1], box[0], box[3], box[2]
            
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                      
            image = cv2.putText(
                    image,
                    f"{float(conf[i]):.2f}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,0,0),
                    2,
                    cv2.LINE_AA,
                )
            
            
            # konec castecne prevzate casti
            output_bboxes.append([x_min, y_min, x_max - x_min, y_max-y_min])
            
    return image, nof_detections, output_bboxes


def detect_faces(image):
    """ Pripravi obrazek pro zpracovani detekcni siti OPENVINO a provede detekci

    Args:
        image (np.darray): obrazek v puvodnim rozliseni
        
    Returns:
        bboxes: souradnice tvari (pascalvoc)
        conf: confidence detekci
    """
    image = cv2.resize(image, (640, 640))
    image = np.array(image)
    input_data = np.expand_dims(np.transpose(image, (0, 1, 2)), 0).astype(np.float32)
    result = compiled_model([input_data])[compiled_model.output(0)]
    
   
    bboxes = result[0,0,:, 3:]
    conf = result[0,0,:, 2:3]
    return bboxes, conf


##############
# GUI cast
# mala cast vychazi z tutorialu
# https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1
# https://www.pythonguis.com/faq/pause-a-running-worker-thread/
##############

# tridy pro aktualizaci videa a zpracovani signalu
class WorkerSignals(QObject):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_frame_count = pyqtSignal(int, int)
    change_duration_signal = pyqtSignal(int, int)
    change_fps_signal = pyqtSignal(float, int)
    set_default_fps_signal = pyqtSignal(int)

class JobRunner(QRunnable):

    signals = WorkerSignals()
    
    def __init__(self, path):
        super().__init__()

        self.is_paused = False
        self.is_killed = False
        self.is_bboxes = False
        self.is_frame_changed = False
        self.time_spot = 0
        self.new_val = 0
        self.is_captured = False
        self.use_openvino = True
        self.use_violajones = False
        self.use_mtcnn = False
        self.path = path # cesta k souboru videa
        self.threshold = 0.3

    @pyqtSlot()
    def run(self):    
        if self.use_openvino == False: # nacteni tf modelu
            _model_list = get_models()
            tf_model = tf.saved_model.load(f"models/{_model_list[model_tf_id]}/default/saved_model")
        
        
        # zpracovani nazvu souboru z cesty pro vytvoreni vystupu
        filename = self.path.split("\\")[-1].split(".")[0]
        self.bbox_file = open(f"{filename}_output.txt", "w")
      
        # vystupni soubor s bboxy ve formatu COCO
        try:
            self.gt_file = open(f"{filename}_gt.txt", "r")
        except Exception:
            self.gt_file = None
        
        self.cap = cv2.VideoCapture(self.path)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fps_count = self.cap.get(cv2.CAP_PROP_FPS)
        try:
            self.signals.set_default_fps_signal.emit(int(fps_count))
        except RuntimeError:
            pass
        
        try:
            duration_s = int(frame_count / fps_count) # v sekundach
        except ZeroDivisionError:
            duration_s = 0
            fps_count = 1
       
        current_frame_number = 0
        
        while self.cap.isOpened():
            if self.is_frame_changed: # skok ve videu pomoci slideru
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.time_spot * self.cap.get(cv2.CAP_PROP_FPS)))
                self.is_frame_changed = False
                
            if self.new_val != 0: # skok ve videu
                current_frame_number += int(self.new_val *self.cap.get(cv2.CAP_PROP_FPS))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
                self.new_val = 0
            
            ret, image = self.cap.read()
            if ret:
                
                if self.bbox_file is not None:
                    self.bbox_file.write(f"frame - {current_frame_number}\n")
                
                if(self.use_violajones):
                    frame_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    frame_gray = cv2.equalizeHist(frame_gray)
                    
                    out_bboxes = []
                    start = time.time()
                    faces = face_cascade.detectMultiScale(frame_gray)

                    for (x,y,w,h) in faces:
                        image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
                        out_bboxes.append([x,y,w,h])
                    end = time.time()
                    nof_detections = len(faces)
                    
                elif(self.use_mtcnn):
                    out_bboxes = []
                    start = time.time()
                    res = mtcnn_detector.detect_faces(image)
                    for r in res:
                        box = r["box"]
                        image = cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,0,255), 2)
                        out_bboxes.append([box[0], box[1], box[2], box[3]])
                    
                    end = time.time()
                    nof_detections = len(res)
                        
                else:
                    # provedeni detekce
                    if(self.use_openvino):
                        start = time.time()
                        bboxes, conf = detect_faces(image)
                        image, nof_detections, out_bboxes = apply_bboxes(image, bboxes, conf, threshold=self.threshold)
                        end = time.time()
                            
                    else:
                        start = time.time()
                        bboxes, conf = detect_faces_tf(tf_model, image)
                        image, nof_detections, out_bboxes = apply_bboxes_tf(image, bboxes.numpy(), conf.numpy(), threshold=self.threshold)
                        end = time.time()
                    
                    
                if self.gt_file is not None:                            
                    try:
                        self.gt_file.readline()
                        count = int(self.gt_file.readline())
                        if count == 0: 
                            self.gt_file.readline()
                        else:
                            for i in range(count):
                                bbox = np.array(self.gt_file.readline().split()[:4], dtype=np.float32)
                                # prevod do pascal VOC
                                bbox[2] = bbox[2] + bbox[0]
                                bbox[3] = bbox[3] + bbox[1]
                                bbox = bbox.astype(int)
                                # zobrazeni GT boxu
                                if self.is_bboxes:
                                    image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                    except Exception:
                        pass
                    
                # zapis do souboru s bboxy    
                self.bbox_file.write(f"{nof_detections}\n")
                for q in range(nof_detections):
                    self.bbox_file.write(f"{' '.join(map(str, out_bboxes[q]))}\n")
                    
                # aktualizace FPS ukazatele
                try:
                    self.signals.change_fps_signal.emit(end-start, nof_detections)
                except RuntimeError:
                    pass
                   
        
                current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                current_duration = int(current_frame_number / fps_count)
                
                # aktualizace dalsich ukazatelu
                try:
                    self.signals.change_pixmap_signal.emit(image)
                    self.signals.change_frame_count.emit(current_frame_number, frame_count)
                    self.signals.change_duration_signal.emit(int(current_duration), duration_s)
                except RuntimeError:
                    pass
                                
                if self.is_captured:
                    cv2.imwrite(str(current_frame_number)+".jpg", image)
                    self.is_captured = False
                
            while self.is_paused:
                if self.is_killed:
                    break
                time.sleep(0)

            if self.is_killed:
                break
                
                
        # ukonceni prace vlakna
        self.cap.release()
        if self.gt_file is not None:
            self.gt_file.close()
        self.bbox_file.close()
           
    # pomocne funkce pro ovladani prehravani a detekce
    def pause(self, play_state):
        if play_state == True:
            self.is_paused = False
        else:
            self.is_paused = True

    def kill(self):
        self.is_killed = True
        
    def show_bboxes(self, show_state):
        if show_state == True:
            self.is_bboxes = True
        else:
            self.is_bboxes = False
            
    def change_frame(self, time_s):
        self.is_bboxes = False
        self.is_frame_changed = True
        self.time_spot = time_s
        
    def change_confidence(self, val):
        self.threshold = float(val/100)

    def update_time(self, new_val):
        self.is_bboxes = False
        self.new_val = new_val
        
    def capture(self):
        self.is_captured = True
        
    def use_ov(self, val):
        self.use_openvino = val
        
    def use_vj(self, val):
        self.use_violajones = val
        
    def use_mtcnn_det(self, val):
        self.use_mtcnn = val

# hlavni GUI
class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        
        self.setMinimumWidth(1050)
        self.setWindowTitle("Face detector in low light conditions")
        self.setWindowIcon(QIcon('icon.png'))
        # zobrazovaci plocha videa (16:9)
        self.video_width = 896
        self.video_height = 504
        self.image_label = QLabel("No video loaded - use Browse button to load")
        self.image_label.setStyleSheet("font-size: 20px")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.resize(self.video_width, self.video_height)
        self.image_label.setMinimumHeight(self.video_height)
        self.image_label.setMinimumWidth(self.video_width)
        
        # slider pro confidence threshold
        self.thresh_slider = QSlider(orientation=Qt.Orientation.Vertical)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setSingleStep(5)
        self.thresh_slider.setValue(30)
        self.thresh_slider.setTickPosition(QSlider.TickPosition.TicksLeft)
        self.thresh_slider.valueChanged.connect(self.thresh_slider_moved)
        
        
        self.thresh_label = QLabel("30 %")
        self.thresh_label.setStyleSheet("font-weight: bold")
        self.thresh_label.setFixedWidth(50)
        
        thresh_layout = QVBoxLayout()
        thresh_layout.addWidget(self.thresh_label)
        thresh_layout.addWidget(self.thresh_slider)
        thresh_layout.setContentsMargins(0, 20, 0, 40)
        
        
        display_layout = QHBoxLayout()
        display_layout.addWidget(self.image_label)
        display_layout.addLayout(thresh_layout)
        
        # horni radek pro otevreni souboru + nastaveni zarizeni
        top_layout = QHBoxLayout()
        self.name = QLabel("Filename")
        self.file_browser_btn = QPushButton("Browse")
        self.file_browser_btn.pressed.connect(self.open_file_dialog)
        self.file_browser_btn.setFixedWidth(75)
        
        self.model_cb = QComboBox()
        self.model_cb.addItems(get_models())
        self.model_cb.currentIndexChanged.connect(self.select_model)

        
        self.device_cb = QComboBox()
        device_list = [d["full_name"] for d in get_devices()]
        self.device_cb.addItems(device_list)
        self.device_cb.currentIndexChanged.connect(self.select_device)
        
        self.ov_checkbox = QCheckBox("OpenVINO?")
        self.ov_checkbox.setChecked(True)
        self.ov_checkbox.toggled.connect(self.set_ov)

        self.qz_checkbox = QCheckBox("Quantization?") 
        self.qz_checkbox.toggled.connect(self.set_qz)     
        
        self.file_text_label = QLabel("File: ")
        self.file_text_label.setFixedWidth(30)
        top_layout.addWidget(self.file_text_label)
        top_layout.addWidget(self.name)
        top_layout.addWidget(self.file_browser_btn)
        top_layout.addStretch()
        top_layout.addWidget(self.model_cb)
        top_layout.addWidget(self.device_cb)
        top_layout.addWidget(self.ov_checkbox)
        top_layout.addWidget(self.qz_checkbox)
        
        # slider
        self.end_value = 0
        self.slider = QSlider(orientation=Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.end_value)
        self.slider.valueChanged.connect(self.slider_moved)
        self.slider.sliderPressed.connect(self.slider_pressed)
        
        self.slider.setEnabled(False)
        
        # radek dat
        data_layout = QHBoxLayout()
        
        frame_layout = QVBoxLayout()
        top_frame_counter_label = QLabel("Frame:")
        top_frame_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_frame_counter_label.setStyleSheet("font-size: 18px")
        self.frame_counter_label = QLabel("0/0")
        self.frame_counter_label.setStyleSheet("font-size: 18px")
        self.frame_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.addWidget(top_frame_counter_label)
        frame_layout.addWidget(self.frame_counter_label)
        
        data_layout.addLayout(frame_layout)
        
        duration_layout = QVBoxLayout()
        top_duration_counter_label = QLabel("Duration:")
        top_duration_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_duration_counter_label.setStyleSheet("font-size: 18px")
        self.duration_label = QLabel("00:00 - 00:00")
        self.duration_label.setStyleSheet("font-size: 18px")
        self.duration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        duration_layout.addWidget(top_duration_counter_label)
        duration_layout.addWidget(self.duration_label)
        
        data_layout.addLayout(duration_layout)
        
        # radek FPS
        fps_layout = QVBoxLayout()
        fps_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fps_label = QLabel("Current: 0 FPS")
        fps_layout.addWidget(self.fps_label)
        self.fps_avg_label = QLabel("Average: 0 FPS")
        self.default_fps = QLabel("Default: 0 FPS")
        fps_layout.addWidget(self.fps_avg_label)
        fps_layout.addWidget(self.default_fps)
        
        self.fps_data_counter = 0
        self.fps_data = 0
        
        data_layout.addLayout(fps_layout)
        
        
        # spodni radek ovladani 1
        control_layout = QHBoxLayout()
        self.btn_pause_toggle = QPushButton("Pause")
        self.btn_skip = QPushButton(">> (5 s)")
        self.btn_rewind= QPushButton("<< (5 s)")
        
        # urcuje stav tlacitka play/pause
        self.play_state = False
        self.btn_pause_toggle.pressed.connect(self.toggle_pause)
        self.btn_rewind.pressed.connect(self.rewind)
        self.btn_skip.pressed.connect(self.skip)
        
        self.btn_pause_toggle.setEnabled(False)
        self.btn_rewind.setEnabled(False)
        self.btn_skip.setEnabled(False)
        
        control_layout.addWidget(self.btn_pause_toggle)
        control_layout.addWidget(self.btn_rewind)
        control_layout.addWidget(self.btn_skip)
        
        # spodni radek ovladani 2
        control_layout2 = QHBoxLayout()
        self.btn_bboxes_toggle = QPushButton("Show GT bboxes")
        self.btn_capture = QPushButton("Capture")
        
        # urcuje stav tlacitka show/hide bboxes
        self.show_bboxes_state = False
        self.btn_bboxes_toggle.pressed.connect(self.toggle_bboxes)
        
        self.btn_capture.pressed.connect(self.capture)
        
        self.btn_bboxes_toggle.setEnabled(False)
        self.btn_capture.setEnabled(False)
        
        control_layout2.addWidget(self.btn_bboxes_toggle)
        control_layout2.addWidget(self.btn_capture)
        
        # hlavni layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        main_layout.addLayout(top_layout)
        main_layout.addLayout(display_layout)
        main_layout.addWidget(self.slider)
        main_layout.addLayout(data_layout)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(control_layout2)
        
        
        self.select_model(0)
        self.select_device(0)
        
        # vystupni csv soubory pro dalsi zpracovani
        self.fps_csv = None
        self.fps_writer = None
        
        self.detections_csv = None
        self.detections_writer = None
        
        self.runner = None
        
        self.vj_state = False
        self.mtcnn_state = False

        self.show()
        
        
    def set_qz(self):
        """Zapina a vypina kvantovani
        """
        self.select_model(self.model_cb.currentIndex())
    
    def set_ov(self):
        """Zapina a vypina OpenVINO
        """
        self.select_model(self.model_cb.currentIndex())
        
        if(self.ov_checkbox.isChecked() == False):
            self.device_cb.setEnabled(False)
            self.qz_checkbox.setEnabled(False)
        else:
            self.device_cb.setEnabled(True)
            self.qz_checkbox.setEnabled(True)
        
        
    def select_model(self, idx):
        """Nastavuje model podle zvoleneho indexu v comboboxu

        Args:
            idx (int): index v comboboxu modelu
        """
        global model
        global model_tf_id
        model_list = get_models()
        
        if model_list[idx] == "Viola-Jones":
            self.vj_state = True
            self.mtcnn_state = False
            self.ov_checkbox.setEnabled(False)
            self.qz_checkbox.setEnabled(False)
            self.device_cb.setEnabled(False)
            
        elif model_list[idx] == "MTCNN":
            self.vj_state = False
            self.mtcnn_state = True
            self.ov_checkbox.setEnabled(False)
            self.qz_checkbox.setEnabled(False)
            self.device_cb.setEnabled(False)
            
        else:
            self.ov_checkbox.setEnabled(True)
            self.qz_checkbox.setEnabled(True)
            self.vj_state = False
            self.mtcnn_state = False
            # openvino
            if self.ov_checkbox.isChecked():
                self.device_cb.setEnabled(True)
                if self.qz_checkbox.isChecked(): # kvantovani on/off
                    path = f"models/{model_list[idx]}/quantized/saved_model.xml"
                else:
                    path = f"models/{model_list[idx]}/accelerated/saved_model.xml"
                                
                model = ie.read_model(model=path)
                self.select_device(self.device_cb.currentIndex())
                
            #tf
            else:                
                model_tf_id = idx
                
            
        
    def select_device(self, idx):
        """Nastavuje zarizeni podle zvoleneho indexu v comboboxu

        Args:
            idx (int): index v comboboxu zarizeni
        """
        global compiled_model
        global model
        device_list = get_devices()
        
        if device_list[idx]["type"] == "MYRIAD":
            self.qz_checkbox.setChecked(False)
            self.qz_checkbox.setEnabled(False)
        else:
            self.qz_checkbox.setEnabled(True)
        
        compiled_model = ie.compile_model(model=model, device_name=device_list[idx]["type"])   
        
        
           
        
    def capture(self):
        """Vytvori screenshot z aktualniho framu
        """
        self.btn_capture.pressed.connect(self.runner.capture)
        
    def slider_pressed(self):
        """Rucni pohyb sliderem znemozni zobrazovani GT boxu
        """
        self.btn_bboxes_toggle.setEnabled(False)
      
    def slider_moved(self):   
        """Nastavi cas videa dle pohybu slideru
        """
        current = self.slider.value()
        end = self.end_value
        self.duration_label.setText(
            f"{int(current/60):02d}:{int(current%60):02d} - {int(end/60):02d}:{int(end%60):02d}" 
            )
        self.slider.sliderMoved.connect(partial(self.runner.change_frame, current))
        
    def thresh_slider_moved(self):
        """Nastavi prah jistoty (confidence) pro detektor
        """
        
        val = self.thresh_slider.value()
        self.thresh_label.setText(f"{val} %")
        
        if (self.runner != None):
            self.thresh_slider.valueChanged.connect(partial(self.runner.change_confidence, val))
       
        
    def rewind(self):
        """Skok SKIP_VAL sekund zpet
        """
        self.btn_bboxes_toggle.setEnabled(False)
        self.btn_rewind.pressed.connect(partial(self.runner.update_time, -SKIP_VAL))
        
    def skip(self):
        """Skok SKIP_VAL sekund dopredu
        """
        self.btn_bboxes_toggle.setEnabled(False)
        self.btn_skip.pressed.connect(partial(self.runner.update_time, SKIP_VAL))
        
    def toggle_pause(self):
        """Ovladani play/pause
        """
        if self.play_state == True:
            self.btn_pause_toggle.setText("Pause")
            self.play_state = False
        else:
            self.btn_pause_toggle.setText("Play")
            self.play_state = True
            
        self.btn_pause_toggle.pressed.connect(partial(self.runner.pause, self.play_state))
        
    def toggle_bboxes(self):
        """Ovladani zobrazeni/skryti GT boxu
        """
        if self.show_bboxes_state == True:
            self.btn_bboxes_toggle.setText("Hide GT bboxes")
            self.show_bboxes_state = False
        else:
            self.btn_bboxes_toggle.setText("Show GT bboxes")
            self.show_bboxes_state = True
            
        self.btn_bboxes_toggle.pressed.connect(partial(self.runner.show_bboxes, self.show_bboxes_state))

    # destruktor
    def closeEvent(self, event):
        
        if self.ov_checkbox.isEnabled() == False:
            if self.runner is not None:
                self.runner.signals.disconnect()
                self.runner.kill()
                self.threadpool.releaseThread()
        
        if self.fps_csv != None:
            self.fps_csv.close()
            
        if self.detections_csv != None:
            self.detections_csv.close()
        
        event.accept()
        
    def get_config(self, path):
        """Vraci aktualni config pro vystupni soubory

        Args:
            path (str): udava nazev souboru

        Returns:
            str: textovy vystup konfigurace pro prvni radky vystupnic analyzacnich souboru
        """
        output = path
        output += "-"
        output += self.model_cb.currentText() + "-"
        output += self.device_cb.currentText() + "-" if self.ov_checkbox.isChecked() else "TF-"
        output += "OpenVINO" if self.ov_checkbox.isChecked() else "No OpenVINO"
        output += "-Quantizated" if self.qz_checkbox.isChecked() else ""
        return output
        
    # otevreni souboru
    # vychazi z
    # https://www.pythontutorial.net/pyqt/pyqt-qfiledialog/
    def open_file_dialog(self):    
        if self.runner != None:   
            # uzavreni souboru
            
            self.name.setText("Filename")
            self.runner.signals.disconnect()
            self.runner.kill()
            self.threadpool.releaseThread()
            
            if self.fps_csv != None:
                self.fps_csv.close()
            
            if self.detections_csv != None:
                self.detections_csv.close()
                
            # vymazat countery    
            self.fps_data_counter = 0
            self.fps_data = 0
           
            self.file_browser_btn.setText("Browse")
            self.runner = None
            
            if(self.model_cb.currentText() != "Viola-Jones" and self.model_cb.currentText() != "MTCNN"):
                self.ov_checkbox.setEnabled(True)
                self.qz_checkbox.setEnabled(True)
            self.slider.setEnabled(False)
            self.btn_rewind.setEnabled(False)
            self.btn_skip.setEnabled(False)
            self.btn_capture.setEnabled(False)
            self.btn_bboxes_toggle.setEnabled(False)
            self.btn_pause_toggle.setEnabled(False)
            self.model_cb.setEnabled(True)
            
            self.btn_pause_toggle.setText("Pause")
             
            self.image_label.clear()
            self.image_label.setText("No video loaded - use Browse button to load")
           
        else: 
            # otevreni noveho souboru
            dialog = QFileDialog(self)
            # vychozi cesta
            dialog.setDirectory(r'C:\VUTFIT\BP\code\data')
            dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
            dialog.setNameFilter("Videos (*.mp4)")
            dialog.setViewMode(QFileDialog.ViewMode.List)
            
            if dialog.exec():
                filenames = dialog.selectedFiles()
                if filenames:
                    
                    input_path = str(Path(filenames[0]))
                    self.name.setText(input_path)
                    
                    self.threadpool = QThreadPool()
                    
                    pathname = input_path.split("\\")[-1]
                    fname = pathname
                    fname += "-" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
                    
                    # CSV writer FPS
                    self.fps_csv = open("out/" + fname + "-fps.csv", "w", newline='') 
                    self.fps_writer = csv.writer(self.fps_csv, delimiter=";")
                    self.fps_writer.writerow([self.get_config(pathname)])
                    # CSV writer detekce
                    self.detections_csv = open("out/" + fname + "-detections.csv", "w", newline='')
                    self.detections_writer = csv.writer(self.detections_csv, delimiter=";")
                    self.detections_writer.writerow([self.get_config(pathname)])

                
                    self.runner = JobRunner(input_path)
                    self.runner.signals.change_pixmap_signal.connect(self.update_image)
                    self.runner.signals.change_frame_count.connect(self.update_frame_counter)
                    self.runner.signals.change_duration_signal.connect(self.update_duration)
                    self.runner.signals.change_fps_signal.connect(self.update_fps)
                    self.runner.signals.set_default_fps_signal.connect(self.set_default_fps)
                    self.threadpool.start(self.runner)

                    self.btn_pause_toggle.setEnabled(True)
                    self.play_state = False
                    self.btn_pause_toggle.pressed.connect(partial(self.runner.pause, self.play_state))
                                        
                    self.btn_bboxes_toggle.setEnabled(True)
                    self.show_bboxes_state = True
                    self.btn_bboxes_toggle.pressed.connect(partial(self.runner.show_bboxes, self.show_bboxes_state))
                    
                    self.btn_rewind.pressed.connect(partial(self.runner.update_time, -SKIP_VAL))
                    self.btn_skip.pressed.connect(partial(self.runner.update_time, SKIP_VAL))
                    
                    self.btn_capture.pressed.connect(self.runner.capture)
                    
                    self.runner.use_ov(self.ov_checkbox.isChecked())     
                    
                    self.runner.use_vj(self.vj_state)  
                    
                    self.runner.use_mtcnn_det(self.mtcnn_state)   
                    
                    self.thresh_slider.sliderMoved.connect(partial(self.runner.change_confidence, self.thresh_slider.value()))      
                    
                    self.runner.change_confidence(self.thresh_slider.value())
                                 
                    # nastaveni povoleni tlacitek a prepinacu   
                    self.ov_checkbox.setDisabled(True)
                    self.qz_checkbox.setDisabled(True)
                    self.model_cb.setDisabled(True)
                    self.slider.setEnabled(True)
                    self.btn_rewind.setEnabled(True)
                    self.btn_skip.setEnabled(True)
                    self.btn_capture.setEnabled(True)
                    
                    self.file_browser_btn.setText("Close")
                 
                
    # aktualizace framu
    # nasledujici 2 funkce vychazeji z
    # https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Aktualizuje zobrazovany frame

        Args:
            cv_img (np.darray): obrazek pro zobrazeni
        """
        if self.runner == None:
            self.image_label.clear()
            self.image_label.setText("No video loaded - use Browse button to load")
        else:
            qt_img = self.convert_cv_qt(cv_img)
            self.image_label.setPixmap(qt_img)
        
    
    def convert_cv_qt(self, cv_img):
        """Prevede OpenCV obrazek na PixMap format

        Args:
           cv_img (np.darray): obrazek pro zobrazeni

        Returns:
            QPixMap: obrazek pro zobrazeni v image_labelu
        """
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_width, self.video_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    # pomocne funkce pro zobrazovani statistik
    @pyqtSlot(int, int)
    def update_frame_counter(self, current, end):
        text = f"{current}/{end}"
        self.frame_counter_label.setText(text)
        
    @pyqtSlot(int, int)
    def update_duration(self, current, end):
        text = f"{int(current/60):02d}:{int(current%60):02d} - {int(end/60):02d}:{int(end%60):02d}" 
        self.duration_label.setText(text)
        self.end_value = end
        self.slider.setMaximum(end)  
        self.slider.setValue(current) 
        
    @pyqtSlot(float, int)
    def update_fps(self, time_dif, nof_detections):
        try:
            fps = 1/time_dif            
        except ZeroDivisionError:
            fps = 0.0 
            
        self.fps_data += fps
        self.fps_data_counter += 1
        fps_avg = self.fps_data/self.fps_data_counter
        
        text = f"Current: {fps:0.3f} FPS"
        self.fps_label.setText(text)
        avg_text = f"Average: {fps_avg:0.3f} FPS"
        self.fps_avg_label.setText(avg_text)
        
        try:
            self.fps_writer.writerow([self.fps_data_counter, f"{fps}"])
            self.detections_writer.writerow([self.fps_data_counter, f"{nof_detections}"])
        except ValueError:
            pass
        
    @pyqtSlot(int)
    def set_default_fps(self, default_fps):
        default_text = f"Default: {default_fps} FPS"
        self.default_fps.setText(default_text)

# spusteni aplikace
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = MainWindow()
    sys.exit(app.exec())