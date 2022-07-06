import numpy as np
import cv2
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)

args = parser.parse_args()

input_file = args.input
output_folder = args.output


with h5py.File(input_file, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
    for frame, mask in enumerate(data):
        print(frame)
        print(mask.shape)
        white_img = np.ones((1080, 1080), dtype='uint8')*255
        mask = cv2.bitwise_and(white_img, white_img, mask=mask)
        print(mask.dtype)
        #mask *= 255
        cv2.imwrite(f"{output_folder}/{frame}.png", mask)
