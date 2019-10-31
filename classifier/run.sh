#/bin/bash

# directories for ImageNet images
mkdir -p imagenet_extracted/plane imagenet_extracted/car

# directories to split up and balance into training data
mkdir -p data/trainset/plane data/validset/plane
mkdir -p data/trainset/car data/validset/plane

# get relevant class images from imagenet
python extract_from_imagenet.py

# split and balance train/validation data for training classifier
python train_split.py

# directories for saving models and plots for analyzing training performance
mkdir models plots

# train classifier for 100 epochs, report best epoch (use this one for classification)
python main.py