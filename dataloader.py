from collections import namedtuple
from torchvision import transforms
import torch

# data loading and preprocessing functions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

State = namedtuple('State',
                        ('image', 'bbox_observed', 'bbox_true', 'action_history'))

def default_collate(batch):
    states = []
    for item in batch:
        image = item[0]
        action_history = torch.zeros(90)
        h = int(item[1]['annotation']['size']['height'])
        w = int(item[1]['annotation']['size']['width'])
        bbox_observed = (0, 0, int(w/2), int(h/2))
        obj = item[1]['annotation']['object']
        if isinstance(obj, list):
            bbox = obj[0]['bndbox']
        else:
            bbox = obj['bndbox']
        left = int(bbox['xmin'])
        upper = int(bbox['ymin'])
        right = int(bbox['xmax'])
        lower = int(bbox['ymax'])
        bbox_true = (left, upper, right, lower)
        states.append(State(image, bbox_observed, bbox_true, action_history))
    return states

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])

def state_transform(states):
    # return the transformed images and action_history for each state
    img_observed = [state.image.crop(state.bbox_observed) for state in states]
    img_t = torch.stack([transform(img) for img in img_observed]).to(device)
    action_history = torch.stack([state.action_history for state in states]).to(device)
    return img_t, action_history
