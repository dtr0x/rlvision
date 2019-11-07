import torchvision.datasets as dset
import torchvision
from dataloader import *
from reinforcement import classifier

# Collect training images from VOC dataset that belong to relevant classes.
# The image should contain only one instance of the relevant class.
# The total image should not be well-classified, while the cropped bounding box
# of the class object should have a high classification score.

CAR = 'car'
PLANE = 'aeroplane'

car_imgs = 0
plane_imgs = 0

IMG_THRESH = 0.9 # maximum classification score for the total image
OBJ_THRESH = 0.9 # minimum classification score for bounding box object

def parse_data(year):
    global car_imgs, plane_imgs
    VOC = dset.VOCDetection("VOC"+year, year=year, image_set='trainval')
    for img, anns in VOC:
        anns = anns['annotation']
        objs = anns['object']
        if not isinstance(objs, list):
            objs = [objs]
        cat_ids = [obj['name'] for obj in objs]
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
            filename = anns['filename']
            bbox_dict = objs[idx]['bndbox']
            bbox = tuple([int(coord) for coord in bbox_dict.values()])
            img_gt = img.crop(bbox)
            img_out = classifier(transform(img).unsqueeze(0).to(device))
            img_gt_out = classifier(transform(img_gt).unsqueeze(0).to(device))
            img_score = img_out[target].item()
            img_gt_score = img_gt_out[target].item()
            if img_score < IMG_THRESH and img_gt_score >= OBJ_THRESH:
                if is_car:
                    car_imgs += 1
                    img.save("coco_voc_images/car/{}".format(filename))
                if is_plane:
                    plane_imgs += 1
                    img.save("coco_voc_images/plane/{}".format(filename))

parse_data('2007')
parse_data('2012')
print("Car images:", car_imgs)
print("Plane images:", plane_imgs)