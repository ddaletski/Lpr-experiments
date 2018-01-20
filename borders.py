import cv2
import numpy as np
import os
import sys
import matplotlib
from matplotlib import pyplot as plt
import argparse
import tqdm
from multiprocessing import Process, Queue, Value
import time


def to_degrees(angle):
    return int(angle * 180 / np.pi)


def to_radians(angle):
    return angle / 180 * np.pi


def bounded(val, low, high):
    if val < low:
        return low
    elif val > high:
        return high
    else:
        return val


def intersection_cart(line1, line2):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    d1 = (x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)
    d2 = (x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)
    d = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)

    x = d1 / d
    y = d2 / d

    return int(x), int(y)


def intersections(img, topline, bottomline, leftline, rightline):
    points = []
    for l1, l2 in [(topline, leftline),
                   (topline, rightline),
                   (bottomline, rightline),
                   (bottomline, leftline)]:

        point = intersection_cart(l1, l2)
        points.append(point)

    return points


def distance_x(line1, line2):
    x1 = (line1[0][0] + line1[1][0]) // 2
    x2 = (line2[0][0] + line2[1][0]) // 2
    return np.abs(x1 - x2)


def distance_y(line1, line2):
    y1 = (line1[0][1] + line1[1][1]) // 2
    y2 = (line2[0][1] + line2[1][1]) // 2
    return np.abs(y1 - y2)


class BorderFinder():
    def __init__(self, hor_threshold, ver_threshold, hor_angle_delta, ver_angle_delta):
        self._hor_threshold = hor_threshold
        self._ver_threshold = ver_threshold
        self._hor_angle_delta = hor_angle_delta
        self._ver_angle_delta = ver_angle_delta


    def get_lines(self, img, percentage, threshold, min_angle=0.00001, max_angle=np.pi,
                  dilation_kernel=(1, 1), max_angle_diff=2*np.pi, max_len_diff=1000):

        edges = cv2.Canny(img, 150, 220)
        edges = edgesHor = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel))
        height, width = img.shape

        hough_lines = cv2.HoughLines(edges, 1, np.pi/90, threshold, min_theta=min_angle, max_theta=max_angle)
        if hough_lines is None:
            return []

        lines_count = int(np.ceil(percentage * len(hough_lines)))
        hough_lines = hough_lines[:lines_count]

        h = height - 1
        w = width - 1
        lines = []

        for i, (rho, theta) in enumerate(hough_lines[:, 0]):
            sin = np.sin(theta)
            cos = np.cos(theta)
            tan = np.tan(theta)
            y0 = rho * sin
            x0 = rho * cos
            skew = x0*y0 > 0

            x1 = bounded(rho/cos, 0, w)
            y1 = bounded(rho/sin, 0, h)

            x2 = bounded(rho/cos - h*tan, 0, w)
            y2 = bounded(rho/sin - w/tan, 0, h)

            if skew and y2 < y1:
                x1, x2 = x2, x1
                if not skew and y2 > y2:
                    x1, x2 = x2, x1

            length = np.sqrt((y1-y2)**2 + (x1-x2)**2)
            line = [(int(x1), int(y1)), (int(x2), int(y2))], (rho, theta)

            if i == 0:
                etalon_len = length
                etalon_angle = theta

            if np.abs(etalon_angle - theta) > max_angle_diff:
                pass
            elif np.abs(length - etalon_len) > max_len_diff:
                pass
            else:
                good = True
                lines.append(line)

        return lines



    def find_top_and_bottom(self, img):
        height, width = img.shape
        w = width - 1
        h = height - 1
        default_lines = [ ((0, 0), (w, 0)), ((0, h), (w, h)) ]

        hor_lines = self.get_lines(img, 1, int(self._hor_threshold*width),
                                   dilation_kernel=(1, 3),
                                   max_angle_diff=to_radians(self._hor_angle_delta),
                                   max_len_diff=w//8)

        if len(hor_lines) < 2:
            return default_lines

        topline, topline_parametric = min(hor_lines, key=lambda l: l[0][0][1] + l[0][1][1])
        bottomline, bottomline_parametric = max(hor_lines, key=lambda l: l[0][0][1] + l[0][1][1])

        if distance_y(topline, bottomline) < height // 4:
            return default_lines

        return topline, bottomline


    def find_left_and_right(self, img):
        height, width = img.shape
        w = width - 1
        h = height - 1
        default_lines = [ ((0, 0), (0, h)), ((w, 0), (w, h)) ]

        normal_angle = np.pi
        min_angle = normal_angle - to_radians(10)
        max_angle = normal_angle + to_radians(10)

        ver_lines = self.get_lines(img, 1, int(self._ver_threshold * height),
                                   min_angle=min_angle, max_angle=max_angle,
                                   max_angle_diff=to_radians(self._ver_angle_delta),
                                   max_len_diff=height//8,
                                   dilation_kernel=(3, 1))

        if len(ver_lines) < 2:
            return default_lines

        leftline, leftline_parametric = min(ver_lines, key=lambda l: l[0][0][0] + l[0][1][0])
        rightline, rightline_parametric = max(ver_lines, key=lambda l: l[0][0][0] + l[0][1][0])

        sortkey = lambda l: l[1] # sort by Y axis
        leftline = sorted(leftline, key=sortkey)
        rightline = sorted(rightline, key=sortkey)

        if distance_x(leftline, rightline) < (w // 2):
            return default_lines

        return leftline, rightline


    def find_borders(self, img):
        topline, bottomline = self.find_top_and_bottom(img)
        leftline, rightline = self.find_left_and_right(img)
        return topline, bottomline, leftline, rightline


    def restore_perspective(self, img, dst_shape):
        width, height = dst_shape
        topline, bottomline, leftline, rightline = self.find_borders(img)
        points = intersections(img, topline, bottomline, leftline, rightline)

        img_copy = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for i, line in enumerate((topline, bottomline, rightline, leftline)):
            color = (0, 150, 0) if i < 2 else (150, 0, 0)
            cv2.line(img_copy, *line, color, 2)
        for point in points:
            cv2.circle(img_copy, point, 2, (0, 0, 200), -1)

        rect_from = np.array([
            points
        ], dtype=np.float32)
        rect_to = np.array([
            [0, 0], [width, 0], [width, height], [0, height]
        ], dtype=np.float32)

        transform = cv2.getPerspectiveTransform(rect_from, rect_to)

        transformed = cv2.warpPerspective(img, transform, dst_shape)
        return transformed, img_copy


def parse_args():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required args')
    required.add_argument('-i', '--input_dir', help='source directory with plates',
                          type=str, required=True)
    required.add_argument('-o', '--output_dir', help='destination directory',
                          type=str, required=True)
    parser.add_argument('-d', '--dbg_dir', help='directory to save debug info',
                        type=str)
    parser.add_argument('-ht', '--hor_threshold', help='min horizontal line length / img width',
                        type=float, default=0.4)
    parser.add_argument('-vt', '--ver_threshold', help='min vertical line length / img height',
                        type=float, default=0.2)
    parser.add_argument('-hd', '--hor_angle_delta', help='max horizontal line angle difference',
                        type=float, default=3)
    parser.add_argument('-vd', '--ver_angle_delta', help='max vertical line angle difference',
                        type=float, default=5)
    parser.add_argument('-w', '--workers', type=int, default=4)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    finder = BorderFinder(args.hor_threshold, args.ver_threshold,
                         args.hor_angle_delta, args.ver_angle_delta)

    debug = args.dbg_dir is not None
    if debug:
        os.makedirs(args.dbg_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    img_count = len(os.listdir(args.input_dir))

    queue = Queue()
    counter = Value('i', 0)

    for i, plate_file in enumerate(tqdm.tqdm(os.scandir(args.input_dir), total=img_count)):
        img = cv2.imread(plate_file.path, cv2.IMREAD_GRAYSCALE)
        transformed, dbg = finder.restore_perspective(img, (144, 32))

        dst_path = os.path.join(args.output_dir, plate_file.name)
        cv2.imwrite(dst_path, cv2.equalizeHist(transformed))
        if debug:
            dbg_path = os.path.join(args.dbg_dir, plate_file.name)
            cv2.imwrite(dbg_path, dbg)


if __name__ == "__main__":
    main()
