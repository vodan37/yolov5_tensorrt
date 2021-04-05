# yolov5 TensorRT
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
