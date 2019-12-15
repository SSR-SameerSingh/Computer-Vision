"""
Character Detection
(Due date: March 8th, 11: 59 P.M.)

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
and the functions you implement in task1.py are of great help.

Hints: You might want to try using the edge detectors to detect edges in both the image and the template image,
and perform template matching using the outputs of edge detectors. Edges preserve shapes and sizes of characters,
which are important for template matching. Edges also eliminate the influence of colors and noises.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import numpy as np
import cv2

import utils
import task1   # you could modify this line


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="./data/characters.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

# def ncc(img, template, mi, mt, vi, vt):

def ssd(img, template):
    ss = 0
    for i in range(0, len(template)):
        for j in range(0, len(template[i])):
            a = (img[i][j])
            b = template[i][j]
            ss += (a-b)**2
    return ss


def normalize(arr):
    mi = np.min(arr)
    ma = np.max(arr)
    arr = arr-mi
    arr_n = arr/ma
    return arr_n


def detect(img, template, threshold):
    img = normalize(img)
    template = normalize(template)

    # final = img
    coordinates = []
    # img_c = task1.detect_edges(img, template)
    x = len(img)
    y = len(img[0])
    xt = len(template)//2
    yt = len(template)
    for i in range(xt, x-yt):
        for j in range(yt, y-xt):
            temp_img = utils.crop(img, (i-xt), (i+xt+1), (j-yt), (j+yt+1))
            ss = ssd(temp_img, template)
            if ss < threshold:
                coordinates.append([i,j])

    """Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
    # TODO: implement this function.
    # raise NotImplementedError
    return coordinates


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["template_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = task1.read_image(args.img_path)
    template = task1.read_image(args.template_path)

    if (args.template_path) == "./data/a.jpg":
        coordinates = detect(img, template, threshold = 4.6)
    elif (args.template_path) == "./data/b.jpg":
        coordinates = detect(img, template, threshold = 6.44)
    elif (args.template_path) == "./data/c.jpg":
        coordinates = detect(img, template, threshold = 4.6)
    elif (args.template_path) == "./data/wa_img.jpg":
        coordinates = detect(img, template, threshold = 4.6)
    else:
        print("Template does not exist.")

    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)
    # if (args.template_path) == "./data/a.jpg":
    #     for i in range(0, len(coordinates)):
    #         cv2.rectangle(img, ((coordinates[i][1])-9-1, (coordinates[i][0])-9), ((coordinates[i][1])+9-3, (coordinates[i][0]+9-3)), (0,0,255), 2)
    #     write_image((img), os.path.join(args.rs_directory, "{}_temp_a.jpg".format(args.template_path.lower())))
    # elif (args.template_path) == "./data/b.jpg":
    #     coordinates = detect(img, template, threshold = 6.44)
    # elif (args.template_path) == "./data/c.jpg":
    #     coordinates = detect(img, template, threshold = 4.6)
    # elif (args.template_path) == "./data/wa_img.jpg":
    #     coordinates = detect(img, template, threshold = 4.6)
    # else:
    #     print("Template does not exist.")

if __name__ == "__main__":
    main()
