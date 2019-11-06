import torch
from collections import namedtuple
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

State = namedtuple('State',
                        ('image', 'obj_class', 'bbox', 'action_history'))

def default_collate(batch):
    states = []
    for image, obj_class in batch:
        action_history = torch.zeros(90)
        bbox = (0, 0, image.width, image.height)
        states.append(State(image, obj_class, bbox, action_history))
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
    img_observed = [state.image.crop(state.bbox) for state in states]
    img_t = torch.stack([transform(img) for img in img_observed]).to(device)
    action_history = torch.stack([state.action_history for state in states]).to(device)
    return img_t, action_history


