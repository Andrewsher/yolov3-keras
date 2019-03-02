# README

Keras-based YOLO v3, baseline solution for [rebar-detection on Data-Foundation](https://www.datafountain.cn/competitions/332/details/rule). It can also be used for VOC dataset.

Step 1. Install required packages.

``` bash
pip install requirements.txt
```

Step 2. Generate VOC dataset. It would not be necessary if you already have one.

``` bash
python toXML/generate_txt.py
python toXML/main.py
```

Step 3. Gererate augmented dataset with h5py file format. This step contributes to speeding up the training procedure.

``` bash
python generate_h5_data_file.py
```

Step 4. Download YOLO v3 weights file from [YOLO website](https://pjreddie.com/darknet/yolo/), and convert it to h5 file format.

``` bash
mkdir pretrined_weights
python convert2keras_weights.py yolov3.cfg yolov3.weights pretrined_weights/yolo.h5
```

Step 5. Train the model, and fine-tune if needed.

``` bash
mkdir model_data_stage_1 model_data_stage_2
python stage_1.py
python stage_2.py
```

Step 6. Generate the predicted result.

``` bash
python predict.py
python output.py
```