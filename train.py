from tqdm.auto import tqdm
import os.path
from construct import train_loader, val_loader, test_loader
from torchmetrics.detection import MeanAveragePrecision
from pprint import pprint
import torch
from config import DEVICE, NUM_CLASSES, DATA_DIR
from utils import create_model, save_model
import matplotlib.pyplot as plt

NUM_EPOCHS = 5

def validate(valid_data_loader, model):
    print('Validating')
    metric = MeanAveragePrecision()
    val_list = []
    model.eval()
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        # print([{k: type(v) for k, v in t.items()} for t in targets])
        targets = [{k: v.to(DEVICE).detach() for k, v in t.items()} for t in targets]

        with torch.no_grad():
            preds = model(images)
            metric.update(preds, targets)
            val_list.append(metric.compute()['map_small'])

    return sum(val_list) / len(val_list)


def train(train_data_loader,val_loader,  model, params):
    print('Training')

    train_itr = 0
    train_loss_list = []
    val_list = []
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data


        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        # train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        val_list.append(validate(val_loader, model))
    return train_loss_list/len(train_loss_list), val_list



model = create_model(num_classes=NUM_CLASSES)
model = model.to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]


for epoch in range(NUM_EPOCHS):
    print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
    train_loss, val_metric = train(train_loader, val_loader, model, params)


# save_model(model, os.path.join(DATA_DIR, 'models', 'model_50.pt' ))

print(validate(test_loader, model))


plt.plot(train_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
plt.plot(val_metric)
plt.xlabel('epoch')
plt.ylabel('map')
plt.show()



