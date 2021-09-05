import os
import numpy
import cv2
from random import sample
from PIL import Image


def main():

    path = "dataset"

    # check if path exist
    val_path = os.path.join(path, "val")
    test_path = os.path.join(path, "test")
    checkPath(val_path)
    checkPath(test_path)


    train_path = os.path.join(path, 'train')
    cla_list = os.listdir(train_path)
    for each_class in cla_list:
        # copy path : dataset/val/class_name ; dataset/test/class_name
        checkPath(os.path.join(val_path, each_class))
        checkPath(os.path.join(test_path, each_class))
        train_img_path = os.path.join(train_path, each_class)

        # generate val dataset
        select_val_list = selectImage(os.path.join(train_path, each_class), select_number=50)
        moveImage(image_path=os.path.join(train_path, each_class),
                  move_to_path=os.path.join(val_path, each_class), img_list=select_val_list)
        # generate test dataset
        select_test_list = selectImage(os.path.join(train_path, each_class), select_number=50)
        moveImage(image_path=os.path.join(train_path, each_class),
                  move_to_path=os.path.join(test_path, each_class), img_list=select_test_list)



def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)

def selectImage(image_path,select_number):
    img_list = os.listdir(image_path)
    return sample(img_list, select_number)

def moveImage(image_path, move_to_path, img_list):
    for each_select_img in img_list:
        img_path = os.path.join(image_path, each_select_img)
        img = Image.open(img_path)
        img_save_path = os.path.join(move_to_path, each_select_img)
        img.save(img_save_path)
        os.remove(img_path)





if __name__ == '__main__':
    main()