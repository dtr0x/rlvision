import torch, sys, ast
from SingleClassDetection import *
from dataloader import *

# prepare python dictionary (as text file for manual editing) of resnet classes
# from VOC image ground truth boxes

voc_classes = []
f = open("voc_classes.txt", "r")
lines = f.readlines()
for l in lines:
    k,v = l.split(',')
    voc_classes.append(k)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

f = open("imagenet_classes.txt").read()
imagenet_classes = ast.literal_eval(f)

conv = torchvision.models.resnet50(pretrained=True)#.to(device)
conv.eval()

class_map = {}

for class_name in voc_classes:
    VOCtrain = SingleClassDetection(class_name, 'trainval')
    train_loader = torch.utils.data.DataLoader(VOCtrain, 
        batch_size=40, collate_fn=default_collate)

    classes = {}
    for _, states in enumerate(train_loader):
        img_gt = [s.image.crop(s.bbox_true) for s in states]
        img_t = torch.stack([transform(img) for img in img_gt])#.to(device)
        out = conv(img_t)
        _, index = torch.max(out, 1)
        classes_batch = [imagenet_classes[i.item()] for i in index]
        for c in classes_batch:
            if c in classes.keys():
                classes[c] += 1
            else:
                classes[c] = 1

    class_map[class_name] = sorted(classes.items(), key=lambda x: x[1], reverse=True)

#[imagenet_classes[idx.item()] for idx in index]
#percentage = torch.nn.functional.softmax(out, dim=1)
#percentage.gather(1, index.unsqueeze(1))



