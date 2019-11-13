## Affirmative vision: Using reinforcement learning to improve accuracy in object detection and tracking

Preliminary design of a reinforcement learning based object detector. The goal of the learning agent is to detect objects in a high resolution image, and refine its viewing region to improve the accuracy of detection.

Refer to the main branch "confidence-detection" for the most up to date and complete implementation. Other branches "singleobject" and "multiobject" refer to a closely related but different implementation with a reward system based on intersection-over-union (IoU), the design of which is based on [Active Object Localization with Deep Reinforcement Learning](https://arxiv.org/abs/1511.06015).

## Using this codebase

```
requirements:
- Python 3
- pytorch
- torchvision
- numpy
- PIL
- matplotlib
- cuda
```

### Detecting an object

The primary use of this codebase is to validate the hypothesis that an RL agent can learn to detect an object in an image with a high level of accuracy by iteratively augmenting a bounding box to improve classification score of the detected object. The steps required are as followed:
1. [Clone the repository and download and extract the data to the parent folder](#dataset)
2. [Train the model](#training)
3. [Run evaluation to find the best model and visualize detections on test data](#evaluation)

## Dataset 

## Training

## Evaluation

## Algorithm Details

## Future Improvements
- Adding a region proposal step to first roughly detect areas where an object exists
- Train a classifier on more classes to detect more types of objects. This would lead to less false detections and general unpredicability of the classifier during search
- Manually test the environment, showing the optimal steps that an agent should take. Does there an exist a path to detect the object on all training data, and how well does this generalize. The existance of such cases is crucial to effective training
- Train classifier and DQN end-to-end rather than using a pretrained classifier 



