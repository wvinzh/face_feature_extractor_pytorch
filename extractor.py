from sphereface import net_sphere
from cosface import net
from resnet import resnet
from insightface import model
import torch
import os
from politics_dataset import PoliticalDataset
import torchvision
import numpy as np
import pickle
from transforms import PoliticTransforms


class Extractor():
    def __init__(self, model_name='resnet'):
        assert model_name in ['resnet', 'sphereface',
                              'cosface', 'insightface'], 'No Such Model Name!!!'
        self.init_model(model_name)
        self.use_gpu = torch.cuda.is_available()
    def load_model_dict(self, dict_path):
        model_dict = torch.load(dict_path)
        self.model.load_state_dict(model_dict)
        if self.use_gpu:
            self.model.cuda()
        return self.model

    def init_model(self, model_name='resnet'):
        self.model_name = model_name
        name_model = {
            'resnet': 'pretrained_model/resnet.pth',
            'sphereface': 'pretrained_model/sphereface.pth',
            'cosface': 'pretrained_model/cosface.pth',
            'insightface': 'pretrained_model/insightface.pth'
        }
        # resnet
        if model_name == 'resnet':
            self.model = resnet.resnet50(num_classes=8631, include_top=False)
        # insightface
        elif model_name == 'insightface':
            self.model = model.Backbone(50, 0.6, 'ir_se')
        # sphereface
        elif model_name == 'sphereface':
            self.model = net_sphere.sphere20a(feature=True)
        # cosface
        elif model_name == 'cosface':
            self.model = net.sphere()
        self.model.load_state_dict(torch.load(
            name_model[model_name], map_location='cpu'))
        self.transform = getattr(PoliticTransforms, 'transform_'+model_name)

    def extract(self, from_folder, image_list, to_folder):
        # transform = None
        dataset = PoliticalDataset(
            from_folder, image_list, to_folder, transform=self.transform)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=4)
        self.model.eval()
        count = 0
        for i, data_batch in enumerate(data_loader):
            # print(data_batch[0].size())
            imgs = data_batch[0]
            categories = data_batch[1]
            basenames = data_batch[2]
            labels = data_batch[3]
            if self.use_gpu:
                imgs = imgs.use_gpu()
            # print(imgs.type())
            output = self.model(imgs)
            # print(output.size())
            output = output.data.cpu().numpy()
            for i, feature in enumerate(output):
                dst_path = os.path.join(to_folder, categories[i])
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)
                np.save(os.path.join(dst_path, basenames[i]+'.npy'), feature)
                # print(os.path.join(dst_path,basenames[i]+'.npy'))
            count += len(imgs)
            print(F"Done {count}")


if __name__ == '__main__':
    image_list = "F:/correct_faces/img_list.txt"
    root_folder = "F://TestFaces"
    feature_folder = "F:/VVVV"
    extractor = Extractor('cosface')
    extractor.extract(root_folder,image_list,feature_folder)
