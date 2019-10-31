#/bin/bash

# directories for extracted images
mkdir -p coco_voc_images/plane coco_voc_images/car

python extract_coco_data.py
python extract_voc_data.py
