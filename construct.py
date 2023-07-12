from utils import  MyDetectDataset, collate_fn
from config import config_train, config_val, config_test
from detection_datasets import DetectionDataset
from torch.utils.data import Dataset, DataLoader

d_train = DetectionDataset().from_disk(**config_train)
d_val = DetectionDataset().from_disk(**config_train)
d_test = DetectionDataset().from_disk(**config_train)

train_dataset = MyDetectDataset(d_train)
val_dataset = MyDetectDataset(d_val)
test_dataset = MyDetectDataset(d_test)

train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True, collate_fn=collate_fn)
val_loader =  DataLoader(val_dataset,batch_size=4,shuffle=True, collate_fn=collate_fn)
test_loader =  DataLoader(test_dataset,batch_size=4,shuffle=True, collate_fn=collate_fn)

print(train_dataset[0])