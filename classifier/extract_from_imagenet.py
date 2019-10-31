import os, shutil

imagenet_folder = "ImageNet"
extracted_folder = "imagenet_extracted"

f = open("class_codes.txt", "r")
lines = f.readlines()

plane_classes = []
car_classes = []

current_class = plane_classes

for l in lines:
    if l != "\n":
        wnid,classes = l.split(':')
        current_class.append(wnid)
    else:
        current_class = car_classes

for wnid in plane_classes:
    img_dir = imagenet_folder + "/train/" + wnid
    if os.path.isdir(img_dir):
        for img in os.listdir(img_dir):
            shutil.copy(os.path.join(img_dir, img), extracted_folder + "/plane")
    img_dir = imagenet_folder + "/valid/" + wnid
    if os.path.isdir(img_dir):
        for img in os.listdir(img_dir):
            shutil.copy(os.path.join(img_dir, img), extracted_folder + "/plane")

for wnid in car_classes:
    img_dir = imagenet_folder + "/train/" + wnid
    if os.path.isdir(img_dir):
        for img in os.listdir(img_dir):
            shutil.copy(os.path.join(img_dir, img), extracted_folder + "/car")
    img_dir = imagenet_folder + "/valid/" + wnid
    if os.path.isdir(img_dir):
        for img in os.listdir(img_dir):
            shutil.copy(os.path.join(img_dir, img), extracted_folder + "/car")

