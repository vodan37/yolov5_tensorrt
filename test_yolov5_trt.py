"""
An example that uses TensorRT's Python api to make inferences.
"""
import argparse
import ctypes
import yaml
import os
from pathlib import Path
import time
import numpy as np
import cv2
from utils.yolov5trt import Yolov5TRT
from utils.decor import plot_one_box, increment_path, from_x1y1x2y2_to_yolo


def test_trt(weights,
             lib,
             data,
             source,
             img_size,
             conf_thres,
             iou_thres,
             save_path,
             name,
             show):
    # load custom plugins
    ctypes.CDLL(lib)
    engine_file_path = weights

    # load coco labels
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    categories = data['names']

    # YoLov5TRT instance
    time_start_load = time.time()
    yolov5_wrapper = Yolov5TRT(engine_file_path, img_size, conf_thres, iou_thres)
    print('time load:', time.time() - time_start_load)

    # Create results dirs
    save_run_dir = Path(increment_path(Path(save_path) / name, exist_ok=False))
    save_run_dir.mkdir(exist_ok=True)
    os.mkdir(os.path.join(save_run_dir, 'images'))
    os.mkdir(os.path.join(save_run_dir, 'labels'))

    input_image_paths = [os.path.join(source, f) for f in os.listdir(source) if f.endswith('.jpg')]
    for input_image_path in input_image_paths:
        # print(input_image_path)
        image_raw = cv2.imread(input_image_path)
        img_height, img_width, img_ch = image_raw.shape
        result_boxes, result_scores, result_classid = yolov5_wrapper.infer(input_image_path)

        parent, filename = os.path.split(input_image_path)
        save_name = os.path.join(parent, "output_" + filename)

        # Draw rectangles and labels on the original image + create *.txt annotation
        with open(os.path.join(save_run_dir, 'labels', 'output_' + filename.replace('.jpg', '.txt')), 'w') as output:
            for i in range(len(result_boxes)):
                box = result_boxes[i]
                plot_one_box(
                    box,
                    image_raw,
                    label="{}:{:.2f}".format(
                        categories[int(result_classid[i])], result_scores[i]
                    ),
                )

                yolo_x, yolo_y, yolo_width, yolo_height = from_x1y1x2y2_to_yolo(float(box[0]), float(box[1]),
                                                                                float(box[2]), float(box[3]),
                                                                                img_width, img_height)
                yolo_str = str(int(result_classid[i])) + ' ' + str(float(result_scores[i])) + ' ' \
                           + str(yolo_x) + ' ' + str(yolo_y) + ' ' + str(yolo_width) + ' ' + str(yolo_height)
                output.write(str(yolo_str) + '\n')

        # 　Image show
        if show:
            cv2.imshow(save_name, image_raw)

        # 　Save image
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cv2.imwrite(str(save_run_dir) + '/' + save_name, image_raw)

    print('mean time NN:', np.mean(yolov5_wrapper.times))
    # destroy the instance
    yolov5_wrapper.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str,
    #                     default='weights/yolov5m_640_helm_int8/yolov5m_640_helm_int8.engine', help='model.engine path')
    # parser.add_argument('--lib', type=str,
    #                     default='weights/yolov5m_640_helm_int8/libmyplugins.so', help='lib path(s)')
    parser.add_argument('--weights', type=str,
                        default='weights/yolov5m_640_helm_fp32/yolov5m_640_helm_fp32.engine', help='model.engine path')
    parser.add_argument('--lib', type=str,
                        default='weights/yolov5m_640_helm_fp32/libmyplugins.so', help='lib path(s)')
    parser.add_argument('--data', type=str, default='helm.yaml', help='*.yaml path')
    parser.add_argument('--source', type=str, default='images/', help='path to images')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save-path', nargs='+', type=str,
                        default='test_result/', help='results path(s)')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--show', default=False, action='store_true', help='show results images')
    opt = parser.parse_args()
    print(opt)

    test_trt(opt.weights,
             opt.lib,
             opt.data,
             opt.source,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_path,
             opt.name,
             opt.show
             )
