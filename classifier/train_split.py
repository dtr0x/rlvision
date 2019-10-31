import os, shutil, random

dirname = "imagenet_extracted/plane"
plane_images = os.listdir(dirname)
random.shuffle(plane_images)
train_set = plane_images[:2200]
valid_set = plane_images[2200:2600]
for img in train_set:
    shutil.copy(os.path.join(dirname, img), "data/trainset/plane")
for img in valid_set:
    shutil.copy(os.path.join(dirname, img), "data/validset/plane")

dirname = "imagenet_extracted/car"
car_images = os.listdir(dirname)
random.shuffle(car_images)
train_set = car_images[:2200]
valid_set = car_images[2200:2600]
for img in train_set:
    shutil.copy(os.path.join(dirname, img), "data/trainset/car")
for img in valid_set:
    shutil.copy(os.path.join(dirname, img), "data/validset/car")
