import torch
from collections import namedtuple
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# State tuple:
    # image: PIL image to perform actions
    # bbox_observed: 4-tuple, currently observed bounding box
    # bboxes_ true: 4-tuples of ground truth bounding boxes (contain the objects)
    # action_history: 10x9 array, one-hot encoded actions vector (last 10 actions)
    # n_trigger: number of times the agent used the trigger action on this image
    # start_pos: integer indicating which corner the search should be restarted
State = namedtuple('State',
                        ('image', 'bbox_observed', 'bboxes_true', 'action_history',
                            'n_trigger', 'start_pos'))

def default_collate(batch):
    states = []
    for item in batch:
        image = item[0]
        action_history = torch.zeros(90)
        w = int(item[1]['annotation']['size']['width'])
        h = int(item[1]['annotation']['size']['height'])
        bbox_observed = (0, 0, w, h)
        objs = item[1]['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]
        bboxes_true = []
        for obj in objs:
            bbox = obj['bndbox']
            left = int(bbox['xmin'])
            upper = int(bbox['ymin'])
            right = int(bbox['xmax'])
            lower = int(bbox['ymax'])
            bboxes_true.append((left, upper, right, lower))
        states.append(State(image, bbox_observed, bboxes_true, action_history, 0, 0))
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


