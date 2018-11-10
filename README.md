# Object Detection - YOLO

YOLO KERAS VERSION

## Dataset


```
http://images.cocodataset.org/zips/train2014.zip <= train images
http://images.cocodataset.org/zips/val2014.zip <= validation images
http://images.cocodataset.org/annotations/annotations_trainval2014.zip <= train and validation annotations
http://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.cfg <= pretrained weight
```

#### Folder Structure

      COCO ── ├── annotations              # unzip trainval2014.zip
              ├── images          
                        ├── train2014      # unzip train2014.zip
                        ├── val2014        # unzip val2014.zip


#### Convert COCO to Pascal VOC Annotation
```
python coco2voc.py change_annotations /somepath/COCO train /somepath/dst/train
python coco2voc.py change_annotations /somepath/COCO val /somepath/dst
```

#### Convert pretrained weight (yolov2.weights) to keras version (.h5)
```
python weight_convert.py convert_yolo_weight_keras /somepath/yolov2.weights /somepath/name.h5
```


