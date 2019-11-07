import torchvision.datasets as dset
import torchvision
from dataloader import *
from reinforcement import classifier

# Collect training images from COCO dataset that belong to relevant classes.
# The image should contain only one instance of the relevant class.
# The total image should not be well-classified, while the cropped bounding box
# of the class object should have a high classification score. 

data = dset.CocoDetection(root = 'COCO/train2017', 
                        annFile = 'COCO/annotations/instances_train2017.json')

CAR = 3
PLANE = 5

car_imgs = 0
plane_imgs = 0

IMG_THRESH = 0.9 # maximum classification score for the total image
OBJ_THRESH = 0.9 # minimum classification score for bounding box object

for img, anns in data:
    cat_ids = [ann['category_id'] for ann in anns]
    ncars = cat_ids.count(CAR)
    nplanes = cat_ids.count(PLANE)
    is_car = (ncars == 1 and nplanes == 0)
    is_plane = (ncars == 0 and nplanes == 1)
    if is_car:
        idx = cat_ids.index(CAR)
        target = 0
    elif is_plane:
        idx = cat_ids.index(PLANE)
        target = 1
    if is_car or is_plane:
        ann = anns[idx]
        bbox = [int(coord) for coord in ann['bbox']]
        x1 = bbox[0]
        y1 = bbox[1]
        w = bbox[2]
        h = bbox[3]
        bbox = (x1, y1, x1+w, y1+h)
        img_gt = img.crop(bbox)
        img_out = classifier(transform(img).unsqueeze(0).to(device))
        img_gt_out = classifier(transform(img_gt).unsqueeze(0).to(device))
        img_score = img_out[target].item()
        img_gt_score = img_gt_out[target].item()
        if img_score < IMG_THRESH and img_gt_score >= OBJ_THRESH:
            if is_car:
                car_imgs += 1
                img.save("coco_voc_images/car/{}.jpg".format(ann['id']))
            if is_plane:
                plane_imgs += 1
                img.save("coco_voc_images/plane/{}.jpg".format(ann['id']))

print("Car images:", car_imgs)
print("Plane images:", plane_imgs)
