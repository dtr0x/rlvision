## Affirmative vision: Using reinforcement learning to improve accuracy in object detection and tracking

Preliminary design of a reinforcement learning based object detector. The goal of the learning agent is to detect objects in a high resolution image, and refine its viewing region to improve the accuracy of detection.

Refer to the main branch "confidence-detection" for the most up to date and complete implementation. Other branches "singleobject" and "multiobject" refer to a closely related but different implementation with a reward system based on intersection-over-union (IoU), the design of which is based on [Active Object Localization with Deep Reinforcement Learning](https://arxiv.org/abs/1511.06015).

### Using this codebase

```
requirements:
- Python 3
- pytorch
- torchvision
- numpy
- PIL
- matplotlib
- cuda
- COCO API 
```

### Detecting an object

The primary use of this codebase is to validate the hypothesis that an RL agent can learn to detect an object in an image with a high level of accuracy by iteratively augmenting a bounding box to improve classification score of the detected object. The steps required are as followed:
1. [Clone the repository and download and extract the data to the parent folder](#dataset)
2. [Train the model](#training)
3. [Run evaluation to find the best model and visualize detections on test data](#evaluation)

### Dataset 
The data file [data.tar.gz](https://drive.google.com/open?id=1CdHEHihc7rzccHX3PDd_HXf2PV2AH4Ra) contains relevant files for training and testing. Included are the COCO, VOC2007, and VOC2012 datasets used for object detection, as well as the coco API files. The folder ```coco_voc_images``` contains images that are preprocessed for training by ```extract_data.sh``` (which simply runs ```extract_coco_data.py``` and ```extract_voc_data.py```). The resulting dataset contains images from COCO and VOC which contain a single object instance from either an ```airplane``` or ```car``` class. Additionally, we verify that the region within the bounding box of the object is classified with a confidence score above a certain threshold (0.9), so that the goal of searching is well-defined for the RL agent. The pretrained car/plane binary classifier (based on Resnet-18) is provided in the ```classifier``` folder, and trained on relevant car/plane classes from ImageNet. 

### Training
```python train.py``` will run the training procedure from the terminal. The ```screen``` program in Linux can be used to offload the training script to the background. During training, A deep Q-network (DQN) is optimized over time to determine the best localizing action that can be taken given an image and observable bounding box region. An object localization search is performed on each image in the ```coco_voc_images``` folder, with the RL agent learning to follow improvements in confidence score. A localization for each image is let to run for a maximum of 40 actions. the training is performed for 100 epochs, with a single epoch consisting of a localization search for each image in the dataset. A DQN model is saved every 5 epochs in the ```models``` folder. 

### Evaluation
Evaluation of all trained DQNs in the ```models``` folder is performed by evaluating recall, which in this case is the percentage of test images with which a DQN successfully localizes the object. In addition, the best model is selected to produce visualizations of the localization process on the test data. Evaluation is run from the terminal by ```python evaluate.py``` and uses functions from ```visualization.py``` to save output to visualization success and failure folders. 

### Future Improvements
- Adding a region proposal step to first roughly detect areas where an object exists. This is most likely required for high-resolution images that contain small objects.
- Train a classifier on more classes to detect more types of objects. This would lead to less false detections and unpredictability of the classifier during search.
- Manually test the environment, showing the optimal steps that an agent should take. Does there an exist a path to detect the object on all training data, and how well does this generalize. The existance of such cases is crucial to effective training
- Train classifier and DQN end-to-end rather than using a pretrained classifier
- Implement more state-of-art RL algorithm for more efficient training and accurate models
- Move all numeric values to argument list
- Move all utility functions to util.py
- Move code in dataloader.py to DataLoader class



