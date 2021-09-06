from model import GoogLeNet
from torchvision import transforms,datasets
import torch
import json
import os
from tqdm import tqdm

def main():
    device = torch.device("cuda：0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    json_file = open("class_indices.json", "r")
    cla_dict = json.load(json_file)

    image_path = "ImageSet"
    batch_size = 32
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                         transform=data_transform)

    test_num = len(test_dataset)


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)




    model = GoogLeNet(num_classes=30, aux_logits=False).to(device)

    model.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for test_data in test_bar:
            test_images, test_labels = test_data
            outputs = model(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

    val_accurate = acc / test_num
    print(val_accurate)