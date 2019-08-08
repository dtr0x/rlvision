from dataloader import *
from torch.utils.data import Dataset
import torchvision

# Plane images from VOC2012 with a bounding box size <= 20% of whole image

class PlaneDetection(Dataset):
    def __init__(self, mode='train'):
        super(PlaneDetection, self).__init__()
        VOC = torchvision.datasets.VOCDetection("VOC2012", image_set=mode)
        self.plane_data = []
        for item in VOC:
            obj = item[1]['annotation']['object']
            if not isinstance(obj, list) and obj['name'] == 'aeroplane':
            	w_img = int(item[1]['annotation']['size']['width'])
            	h_img = int(item[1]['annotation']['size']['height'])
            	area_img = w_img * h_img

            	bbox = item[1]['annotation']['object']['bndbox']
            	bbox = [int(val) for val in bbox.values()]

            	w_box = bbox[2] - bbox[0]
            	h_box = bbox[3] - bbox[1]
            	area_box = w_box * h_box

            	if area_box/area_img <= 0.2:
                    self.plane_data.append(item)

    def __getitem__(self, index):
        return self.plane_data[index]

    def __len__(self):
        return len(self.plane_data)
