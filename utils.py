from torch.utils.data import Dataset
from config import SCALE
import torchvision.transforms as transforms
import torchvision
from torchvision.io import read_image
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
class MyDetectDataset(Dataset):
    def __init__(self, dd,  size_transform =SCALE):
        self.size_transform = size_transform
        self.transforms = transforms.Compose([transforms.Resize((self.size_transform, self.size_transform))])
        self.size_transform = size_transform
        self.all_images = dd.data['image_path'].values
        target = {}
        target['image_id'] = dd.data.index.values
        target["labels"] = dd.data['category_id'].values
        target["area"] = dd.data['area'].values
        all_llists = []
        num = 0
        for j in dd.data['bbox']:
            img_id = dd.data.index[num]
            llist = []
            for k in j:
                temp = k.bbox
                temp[0] = temp[0] / dd.data['width'][num] * SCALE
                temp[2] = temp[2] / dd.data['width'][num] * SCALE
                temp[1] = temp[1] / dd.data['height'][num] * SCALE
                temp[3] = temp[3] / dd.data['height'][num] * SCALE
                llist.append(temp)
            num += 1
            all_llists.append(llist)
        target['boxes'] = all_llists
        for k in range(len(all_llists)):
            temp = []
            for j in (all_llists[k]):
                temp.append((j[2] - j[0]) * (j[3] - j[1]))
            target['area'][k] = temp
        self.target = target


    def __getitem__(self, idx):

        image_path = self.all_images[idx]
        image = read_image(image_path, mode=torchvision.io.image.ImageReadMode.RGB)
        image = self.transforms(image)/255.0


        keys = self.target.keys()
        values = [self.target[k][idx]for k in keys]
        target_idx= dict(zip(keys, values))
        target_idx['labels'] = torch.tensor(target_idx['labels'])
        target_idx['boxes'] = torch.tensor(target_idx['boxes']).reshape(-1, 4)
        target_idx['image_id'] = torch.tensor(target_idx['image_id'])
        target_idx['iscrowd'] = torch.tensor([0]*target_idx['boxes'].shape[0])
        target_idx['area'] = torch.tensor(target_idx['area'])
        print(target_idx)
        return image, target_idx

    def __len__(self):
        return len(self.all_images)


def collate_fn(batch):
    return tuple(zip(*batch))



def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def save_model(model, filepath):
    with open(filepath, 'wb') as file:
        torch.save(model, file)

