import glob
import re
import random
import cv2
from pathlib import Path


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def from_x1y1x2y2_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """
    Description: Conver bbox format to yolo format
    return:
        x_centre, y_centre, width, height.
        float values relative to width and height of image, it can be equal from (0.0 to 1.0]
    """
    width_rastr = x2 - x1
    height_rastr = y2 - y1
    x_centre_rastr = x2 - width_rastr / 2
    y_centre_rastr = y2 - width_rastr / 2
    return x_centre_rastr / img_width, y_centre_rastr / img_height, width_rastr / img_width, height_rastr / img_height



def get_bound_from_yolo(image_resolution, y_x, y_y, y_w, y_h):
    """
    Conver bounder yolo format to pixels format
    return x_top_lef, y_top_lef, x_bot_right, y_bot_right
    """
    img_H = image_resolution[0]
    img_W = image_resolution[1]
    x1 = int(y_x * img_W - (y_w * img_W / 2.0))
    x2 = int(y_x * img_W + (y_w * img_W / 2.0))
    y1 = int(y_y * img_H - (y_h * img_H / 2.0))
    y2 = int(y_y * img_H + (y_h * img_H / 2.0))
    return x1, y1, x2, y2