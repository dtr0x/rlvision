from dataloader import *
from torch.utils.data import Dataset
import torchvision

# Plane images from VOC2012

def get_bbox_percentage(item):
    obj = item[1]['annotation']['object']
    if not isinstance(obj, list) and obj['name'] == 'aeroplane':
        w_img = int(item[1]['annotation']['size']['width'])
        h_img = int(item[1]['annotation']['size']['height'])
        area_img = w_img*h_img
        bbox = item[1]['annotation']['object']['bndbox']
        left = int(bbox['xmin'])
        upper = int(bbox['ymin'])
        right = int(bbox['xmax'])
        lower = int(bbox['ymax'])
        w_box = right - left
        h_box = lower - upper
        area_box = w_box * h_box
        return area_box/area_img

class PlaneDetection(Dataset):
    def __init__(self, mode='train'):
        super(PlaneDetection, self).__init__()
        VOC = torchvision.datasets.VOCDetection("VOC2012", image_set=mode)
        self.plane_data = []
        for item in VOC:
            ap = get_bbox_percentage(item)
            if ap is not None and ap <= 0.5:
                self.plane_data.append(item)

    def __getitem__(self, index):
        return self.plane_data[index]

    def __len__(self):
        return len(self.plane_data)
