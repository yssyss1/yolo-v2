# Object Detection - YOLO

YOLO KERAS VERSION

## To run
```
1. 데이터와 weight 파일들을 다운받는다 (1)
2. 데이터들을 파일구조에 맞춰서 이동시킨다 (1.1)
3. 데이터들을 pascal voc 형식으로 바꾼다 (1.2)
    - 2번에서 이동시킨 데이터 경로를 change_annotations 인자들로 넣어준다
4. 다운받은 weight를 케라스 형식으로 바꾼다 (1.3)
    - 1번에서 다운받은 pretrained weight 경로를 convert_yolo_weight_keras 인자로 넣어준
5. config/yolo.json을 작성한다 (2.1)
    - 대부분은 수정할 필요없고 배치 사이즈, lr 등 기본적인 Hyperparameter들을 수정한다
6. 학습한다 (2.2)
    - config_path를 인자로 넣어줘야 함. default 값으로 './config/yolo.json'을 받도록 되어있음
7. Inference (2.3)
    - weight_path, image_path 명시해줘야 함
    - config_path, obj_threshold, nms_threshold는 default 값들을 받도록 되어있음 - optional
    - 각 default 값들을 config_path: './config/yolo.json' obj_threshold: 0.3, nms_threshold: 0.3

3, 4, 6, 7번에서 함수를 실행시킬 때 인자들의 의미를 잘 모르겠다면 
python [파일 이름.py] [함수 이름] -h 를 실행하길 바람
ex) python yolo_main.py inference -h

```

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
python coco2voc.py change_annotations /somepath/COCO train /somepath/저장경로
python coco2voc.py change_annotations /somepath/COCO val /somepath/저장경로
```

### 1.3 Convert pretrained weight (yolov2.weights) to keras version (.h5)

```
python weight_convert.py convert_yolo_weight_keras /somepath/yolov2.weights /somepath/저장경
```

## 2. Training & Inference

### 2.1 Set configuration file 
```
You have to specify hyperparameters and constants in .config/yolo.json
This file will be used for training and inference
```

### 2.2 Training
```
python yolo_main.py training ./config/yolo.json
```

### 2.3 Inference
```
python yolo_main.py inference weight_path image_path config_path[optional] obj_threshold[optional] nms_threshold[optional]
```