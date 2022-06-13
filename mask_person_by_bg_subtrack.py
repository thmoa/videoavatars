import cv2
import numpy as np

vdo_file = "E:/CMU Med Data Analysis/3d_reconstruck/dataset/input/Noratap/noratap_360.mp4"
bg_file = "E:/CMU Med Data Analysis/3d_reconstruck/dataset/input/Noratap/bg.jpg"

mask_out_path = "E:/CMU Med Data Analysis/3d_reconstruck/dataset/input/Noratap/mask"

# Set Up Vdo Write
vdo_name = "E:/CMU Med Data Analysis/3d_reconstruck/dataset/input/Noratap/noratap_360_1080.avi"
print("VDO Save ", vdo_name)
vdo_out = cv2.VideoWriter(vdo_name, cv2.VideoWriter_fourcc(
    *'DIVX'), 25, (1080, 1080))

cap = cv2.VideoCapture(vdo_file)
img_bg = cv2.imread(bg_file)

w = img_bg.shape[0]
h = img_bg.shape[1]

target_size = 720

s_x = (w//2-target_size//2) + 350
e_x = (w//2+target_size//2) + 350
s_y = 0
e_y = target_size

img_bg = img_bg[s_y:e_y, s_x:e_x]

frame = 0

if (cap.isOpened() == False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:

        img = img[s_y:e_y, s_x:e_x]

        cv2.imshow('Img', img)

        # bg Subtrack
        img_mask = cv2.subtract(img_bg, img)
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)

       # img_mask = cv2.equalizeHist(img_mask)

        ret_th, img_mask = cv2.threshold(img_mask, 30, 255, cv2.THRESH_BINARY)

        # img_mask = cv2.dilate(img_mask, np.ones((10, 10)))

        cv2.imshow('Mask', img_mask)

        img_mask = cv2.resize(img_mask, (1080, 1080))
        img = cv2.resize(img, (1080, 1080))

        # Save Img
        cv2.imwrite(f"{mask_out_path}/{frame}.png", img_mask)
        vdo_out.write(img)

        frame += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

vdo_out.release()
cap.release()
cv2.destroyAllWindows()
