# pca - 2018
# This script generates from an image and a labeled image a dataset of smaller images.

import argparse
import numpy as np
from skimage import io



# calculate the percentage of pixels with this value
def perc_value_in_img(sub_img_label, value):
    w, h = sub_img_label.shape
    total_size = w * h

    return np.count_nonzero(sub_img_label == value) / total_size

# classify the image using the labeled image
def find_class(sub_img_label):
    # value water is (0)
    perc_water = perc_value_in_img(sub_img_label, 0)

    if perc_water >= 0.8:
        return "water"
    if perc_water >= 0.2:
        return "edge"
    return "green"

# walk through the images and generate a sub image.
# the stride is half the image size, so each image partially overlaps.
def generate_images(img_orig, img_labels, n):
    w, h, dim = img_orig.shape

    print("w: {0}  h: {1}".format(w, h))

    if w != img_labels.shape[0] or h != img_labels.shape[1]:
        raise Exception('dimensions have a different shape')

    for idx_w in range(0, w - n, n//2):
        for idx_h in range(0, h - n, n//2):
            yield img_orig[idx_w: idx_w + n, idx_h: idx_h + n, :], \
                  find_class(img_labels[idx_w: idx_w+n, idx_h: idx_h+n])


# generate the dataset and store the images as file on folder 'location'
def generate_dataset(location, img_orig, img_labels, img_size):
    cnt = 0
    for sub_img_orig, img_class in generate_images(img_orig, img_labels, img_size):
        print(img_class)

        # subfolder for each class needs to be there already.
        io.imsave(r'{0}\{1}\img_orig_{2}.jpg'.format(location, img_class, cnt), sub_img_orig)
        cnt += 1


def main(args):
    # generate the dataset

    # Load an color image in grayscale
    img_orig = io.imread(args['image'])
    img_labels = io.imread(args['label'])

    print(img_orig.shape)
    print(img_labels.shape)

    generate_dataset(args['output'], img_orig, img_labels, 50)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    ap.add_argument("-l", "--label", required=True,
                    help="path to label image")
    ap.add_argument("-o", "--output", required=True,
                    help="path to output dataset")

    args = vars(ap.parse_args())

    main(args)
