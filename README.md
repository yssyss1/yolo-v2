# Object Detection - YOLO

YOLO KERAS VERSION

## 1. Dataset

```
http://images.cocodataset.org/zips/train2014.zip <= train images
http://images.cocodataset.org/zips/val2014.zip <= validation images
http://images.cocodataset.org/annotations/annotations_trainval2014.zip <= train and validation annotations
http://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.cfg <= pretrained weight
```

### 1.1 Folder Structure

      COCO ── ├── annotations              # unzip trainval2014.zip
              ├── images          
                        ├── train2014      # unzip train2014.zip
                        ├── val2014        # unzip val2014.zip


### 1.2 Convert COCO to Pascal VOC Annotation

```
python coco2voc.py change_annotations /somepath/COCO train /somepath/dst/train
python coco2voc.py change_annotations /somepath/COCO val /somepath/dst
```

### 1.3 Convert pretrained weight (yolov2.weights) to keras version (.h5)

```
python weight_convert.py convert_yolo_weight_keras /somepath/yolov2.weights /somepath/name.h5
```

## 2. Training & Inference

### 2.1 Set configuration file 
```
You have to specify hyperparameters and constants in .config/yolo.json. 
This file will be used for training and inference.
```

### 2.2 Training
```
python yolo_main.py training ./config/yolo.json
```

### 2.3 Inference
```
python yolo_main.py inference weight_path image_path config_path[optional] obj_threshold[optional] nms_threshold[optional]
```

## To run
```
1. 데이터와 weight 파일들을 다운받는다 (1)
2. 데이터들을 파일구조에 맞춰서 이동시킨다 (1.1)
3. 데이터들을 pascal voc 형식으로 바꾼다 (1.2)
4. 다운받은 weight를 케라스 형식으로 바꾼다 (1.3)
5. config/yolo.json을 작성한다 (2.1)
6. 학습한다 (2.2)
7. Inference (2.3)
```