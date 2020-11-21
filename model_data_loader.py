from torchvision import datasets, transforms
import torch
import json
from PIL import Image

class ModelDataLoader():

    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.train_dir = data_dir + '/train'
        self.valid_dir = data_dir + '/valid'
        self.test_dir = data_dir + '/test'

        normalize_transform = transforms.Normalize(
                                [0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]
                            )


        # image transform for training data
        self.train_transform = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_transform,
            ])

        # image transform for test and validation data
        self.test_transform = transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize_transform,
            ])

    def get_train_data(self):
        train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=64)
        return (train_dataset, train_dataloader)

    def get_validation_data(self):
        valid_dataset = datasets.ImageFolder(self.valid_dir, transform=self.test_transform)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, batch_size=64)
        return (valid_dataset, valid_dataloader)

    def get_test_data(self):
        test_dataset = datasets.ImageFolder(self.test_dir, transform=self.test_transform)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=64)
        return (test_dataloader, test_dataloader)

    def get_label_dict(self, filename):
        with open(filename, 'r') as f:
            cat_to_name = json.load(f)

        return cat_to_name

    def process_image(self, filename):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        # TODO: Process a PIL image for use in a PyTorch model

        image = Image.open(filename)
        return self.test_transform(image)