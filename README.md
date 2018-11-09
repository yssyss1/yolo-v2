# Object Detection - YOLO

YOLO KERAS VERSION

## Dataset


```
http://images.cocodataset.org/zips/train2014.zip <= train images
http://images.cocodataset.org/zips/val2014.zip <= validation images
http://images.cocodataset.org/annotations/annotations_trainval2014.zip <= train and validation anno
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


