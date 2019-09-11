from torch.utils.data import Dataset
import torchvision

class SingleClassDetection(Dataset):
    def __init__(self, class_name, mode='train'):
        super(SingleClassDetection, self).__init__()
        VOC = torchvision.datasets.VOCDetection("VOC2012", image_set=mode)
        self.data = []
        for item in VOC:
            obj = item[1]['annotation']['object']
            if not isinstance(obj, list):
                if obj['name'] == class_name:
                    self.data.append(item)
                elif class_name == 'all':
                    self.data.append(item)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
