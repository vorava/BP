# Detekce obličeje ve zhoršených světelných podmínkách
- Vojtěch Orava (xorava02) 
- BP 2022/2023, vedoucí Ing. Tomáš Goldmann

## Rozložení souborů
```
.
├── text/
│   └── text práce
└── code/
    ├── api/
    │   ├── checkpointy a config SSD+Resnet50
    │   ├── skripty na vytvoření augmentace
    │   ├── skripty na vytvoření tfrecords
    │   └── ssd.ipynb - testovací notebook
    ├── data/
    │   ├── video2.mp4 - testovací video
    │   └── output2.mp4 - testovací video převedené mirnetem
    ├── gui/
    │   ├── models /
    │   │   └── složka s modely (aktuálně jen SSD+Resnet50) - (ne)akcelerovaný, kvantovaný
    │   ├── app.py - gui aplikace
    │   ├── icon.png - ikona aplikace
    │   └── quantization.py - převádí OpenVINO model na kvantovaný model
    ├── mirnet/
    │   ├── convert_to_mirnet.py - převádí video mirnetem
    │   ├── mirnet.ipynb - skript k natrénování Mirnetu
    │   └── myMirnet/
    │       └── natrénovaný mirnet model
    ├── output/
    │   └── get_data.py - vytváří grafy z dat vytvořených GUI aplikací
    ├── evaljob.sh - skript pro evaluaci v Metacentru
    └── job.sh - skript pro trénování v Metacentru

```

## Užitečné a používané příkazy
```
Převedení saved_modelu z TF2 do OpenVINO formátu

mo --saved_model_dir default\saved_model --transformations_config C:\Users\vojta\AppData\Local\Programs\Python\Python38\Lib\site-packages\openvino\tools\mo\front\tf\ssd_support_api_v2.4.json --tensorflow_object_detection_api_pipeline_config default\pipeline.config --input_shape [1,640,640,3] --data_type FP16
```
https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html#custom-input-shape

## Knihovny
- TensorFlow 2
- OpenVINO 2022.2 (pip install openvino-dev==2022.2)
- další viz soubor Seznam příkazů....docx