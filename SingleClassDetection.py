from torch.utils.data import Dataset
import torchvision

class SingleClassDetection(Dataset):
    def __init__(self, class_name, mode='train'):
        super(SingleClassDetection, self).__init__()
        VOC = torchvision.datasets.VOCDetection("VOC2012", image_set=mode)
        self.data = []
        for item in VOC:
            idat = item[1]['annotation']
            if not isinstance(idat['object'], list):
                if idat['object']['name'] == class_name:
                    w_img = int(idat['size']['width'])
                    h_img = int(idat['size']['height'])
                    area_img = w_img * h_img
                    bbox = idat['object']['bndbox']
                    left = int(bbox['xmin'])
                    upper = int(bbox['ymin'])
                    right = int(bbox['xmax'])
                    lower = int(bbox['ymax'])
                    w_box = right - left
                    h_box = lower - upper
                    area_box = w_box * h_box
                    if area_box/area_img <= 0.3:
                        self.data.append(item)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
