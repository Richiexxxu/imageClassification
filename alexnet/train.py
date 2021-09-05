import torch
import torch.nn as nn
from torchvision import transforms, datasets,utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet
import os
import json
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else " cpu")
print(device)


data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}



json_file = open("class_indices.json", "r")
cla_dict = json.load(json_file)


image_path = "ImageSet"
batch_size = 32
train_dataset = datasets.ImageFolder(root = os.path.join(image_path, "train"),
                                     transform=data_transform['train'])
validate_datast = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=data_transform['val'])
train_num = len(train_dataset)

val_num = len(validate_datast)


train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
validate_loader = torch.utils.data.DataLoader(validate_datast,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

net = AlexNet(num_classes=30, init_weights=True)

net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0002)

epoches = 20
saved_path = 'alexNet_weight.pth'
best_acc = 0.0
train_steps = len(train_loader)
for epoch in range(epoches):
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epoches,
                                                                 loss)

    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

    val_accurate = acc / val_num
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, running_loss / train_steps, val_accurate))

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), saved_path)




