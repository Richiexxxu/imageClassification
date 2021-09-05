import os
import json
import torch

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import darknet53


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.106],
                                     std=[0.229, 0.224, 0.225])

    data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    # read class_indices
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file : '{}' does not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indices = json.load(json_file)

    # create mode
    model = darknet53(num_classes=len(class_indices)).to(device)

    # load model
    weights_path = "./new_model_best.pth.tar"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)

    weights_parameters = torch.load(weights_path, map_location=device)['state_dict']
    model.load_state_dict(weights_parameters)

    #prediction
    model.eval()
    with torch.no_grad():
        # load each image
        test_path = "dataset/test"
        cla_list = os.listdir(test_path)
        for each_target_class in cla_list:
            cla_path = os.path.join(test_path, each_target_class)
            for each_predict_img in os.listdir(cla_path):
                # each image prediction
                img_path = os.path.join(cla_path, each_predict_img)
                assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
                img = Image.open(img_path)
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0)

                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()
                print_res = "class: {}   prob: {:.3}".format(class_indices[str(predict_cla)],
                                                             predict[predict_cla].numpy())
                print(print_res + " : " + each_target_class)


if __name__ == '__main__':
    main()