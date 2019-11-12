import torch
from collections import namedtuple
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define the state for RL agent
# image: the complete image used for object detection
# obj_class: the class of the object the image contains (0 for car, 1 for plane)
# bbox: 4-tuple of the bounding box coordinates corresponding to the currently
#       viewed region to of the agent
# action_history: 90-dimensional vector, an encoding of the last 10 actions
State = namedtuple('State',
                        ('image', 'obj_class', 'bbox', 'action_history'))

# prepares the data (transforms images to states)
def default_collate(batch):
    states = []
    for image, obj_class in batch:
        action_history = torch.zeros(90)
        bbox = (0, 0, image.width, image.height)
        states.append(State(image, obj_class, bbox, action_history))
    return states

# default transform, required for classifier based on Resnet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])

# return the transformed images and action_history for a batch of states
# output is used as input to DQN
def state_transform(states):
    img_observed = [state.image.crop(state.bbox) for state in states]
    img_t = torch.stack([transform(img) for img in img_observed]).to(device)
    action_history = torch.stack([state.action_history for state in states]).to(device)
    return img_t, action_history


