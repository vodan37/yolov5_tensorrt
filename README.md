# Yolov5 TensorRT
Implementation yolov5 with [TensorRT](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5)

Install the [dependencies](https://github.com/wang-xinyu/tensorrtx/blob/master/tutorials/install.md)

# Getting started

Prepare you model as in the [example](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5) and put '*.so' and '*.engine' files to dir 'weights'. Or just download my [models](https://www.kaggle.com/vodan37/yolo-helmethead):
```
    sh download_weights.sh
```

Prepare some test images in the 'images' folder. Or download my [images](https://www.kaggle.com/vodan37/yolo-helmethead):
```
    sh download_images.sh
```

If you downloaded my weights you can start with the command:
```
    python test_yolov5_trt.py
```
At the end of the program results will appear in the 'test_results' folder. In the folder 'images' there will be images with drawn bboxes and in the folder 'labels' there will be annotations in yolo format.

Or you can configure programm as you wish:
```
python test_yolov5_trt.py --help
usage: test_yolov5_trt.py [-h] [--weights WEIGHTS] [--lib LIB] [--data DATA]
                      [--source SOURCE] [--img-size IMG_SIZE]
                      [--conf-thres CONF_THRES] [--iou-thres IOU_THRES]
                      [--save-path SAVE_PATH [SAVE_PATH ...]]
                      [--name NAME] [--show]

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS     model.engine path
  --lib LIB             lib path(s)
  --data DATA           *.yaml path
  --source SOURCE       path to images
  --img-size IMG_SIZE   inference size (pixels)
  --conf-thres CONF_THRES
                        object confidence threshold
  --iou-thres IOU_THRES
                        IOU threshold for NMS
  --save-path SAVE_PATH [SAVE_PATH ...]
                        results path(s)
  --name NAME           save results to project/name
      --show                show results images
```
For example:
```
python test_yolov5_trt.py --weights weights/yolov5m_640_helm_fp32/yolov5m_640_helm_fp32.engine # path to *.engine
                `         --lib     weights/yolov5m_640_helm_fp32/libmyplugins.so # path to *.so
                          --data    helm.yaml # path to *.yaml coco format
                          --source  images/   # path to images folder
                          --img-size 640      # NN input image size
                          --conf-thres 0.2    # conf-thres
                          --iou-thres  0.6    # iou-thres
                          --save-path  test_result/     # save path
                          --name       exp              # name folder
                          --show       False            # show or not
```

# Test results of my model
Tests were made by [this project](https://github.com/Cartucho/mAP) with GTX1660S. [Detailed results.](https://www.kaggle.com/vodan37/yolo-helmethead)
Short results:
| Model | size | mAP All | mAP All | Speed
|----------------|:---------:|----------------:|----------------:|
| Vanil YoloV5m | 640 | 0.939 | 59 ms*|
| TRT FP32 | 640 | 0.902 | 25 ms** |
| TRT Int8 | 640 | 0.818 | 17 ms*** |

* - batch-size 32
** - batch-size 1, without preprocess and postprocess 15 ms
*** - batch-size 1, without preprocess and postprocess 7 ms
