import torchvision
import numpy as np

# Record the number of items per class in VOC training data
# that contain a single detectable object per image.

VOC = torchvision.datasets.VOCDetection("VOC2012")

voc_classes = {}

for item in VOC:
    obj = item[1]['annotation']['object']
    if not isinstance(obj, list):
        class_name = obj['name']
        if class_name in voc_classes.keys():
            voc_classes[class_name] += 1
        else:
            voc_classes[class_name] = 1

k = list(voc_classes.keys())
v = list(voc_classes.values())

f = open("voc_classes.txt", "w+")
for i in np.argsort(v)[::-1]:
	f.write("{0},{1}\n".format(k[i], v[i]))
f.close()