import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model import GoogLeNet


def main():
    def main():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(device))

        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            "val": transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

        json_file = open("class_indices.json", "r")
        cla_dict = json.load(json_file)

        data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
        image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
        assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
        train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                             transform=data_transform["train"])
        train_num = len(train_dataset)

        batch_size = 32
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=nw)

        validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                                transform=data_transform["val"])
        val_num = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=batch_size, shuffle=False,
                                                      num_workers=nw)

        print("using {} images for training, {} images for validation.".format(train_num,
                                                                               val_num))


        net = GoogLeNet(num_classes=30, aux_logits=True, init_weights=True)
        net.to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0003)

        epochs = 20
        best_acc = 0.0
        save_path = 'googleNet.pth'
        train_steps = len(train_loader)
        for epoch in range(epochs):
            # train
            net.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader)
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                logits, aux_logits2, aux_logits1 = net(images.to(device))
                loss0 = loss_function(logits, labels.to(device))
                loss1 = loss_function(aux_logits1, labels.to(device))
                loss2 = loss_function(aux_logits2, labels.to(device))
                loss = loss0 + loss1 * 0.3 + loss2 * 0.3
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)

            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(validate_loader)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))  # eval model only have last output layer
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)


    if __name__ == '__main__':
        main()