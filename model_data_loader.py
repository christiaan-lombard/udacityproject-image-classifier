#!/usr/bin/env python3
#
# PROGRAMMER: Christiaan Lombard
# DATE CREATED: 2020-11-21
# REVISED DATE: 2020-11-21
# PURPOSE: Define model data loader
#
#

from torchvision import datasets, transforms
import torch
import json
from PIL import Image


class ModelDataLoader():
    """Loads data and defines required transforms
    """

    def __init__(self, data_dir):
        """Create instance of model data loader

        Args:
            data_dir (str): Root director of image data
        """

        self.data_dir = data_dir
        self.train_dir = data_dir + '/train'
        self.valid_dir = data_dir + '/valid'
        self.test_dir = data_dir + '/test'

        # normalization required by models trained on ImageNet
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
        """Get training dataset and dataloader

        Returns:
            (ImageFolder, DataLoader): The image folder and data loader
        """
        train_dataset = datasets.ImageFolder(
            self.train_dir, transform=self.train_transform)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, batch_size=64)
        return (train_dataset, train_dataloader)

    def get_validation_data(self):
        """Get validation dataset and dataloader

        Returns:
            (ImageFolder, DataLoader): The image folder and data loader
        """
        valid_dataset = datasets.ImageFolder(
            self.valid_dir, transform=self.test_transform)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, shuffle=False, batch_size=64)
        return (valid_dataset, valid_dataloader)

    def get_test_data(self):
        """Get test dataset and dataloader

        Returns:
            (ImageFolder, DataLoader): The image folder and data loader
        """
        test_dataset = datasets.ImageFolder(
            self.test_dir, transform=self.test_transform)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, shuffle=False, batch_size=64)
        return (test_dataloader, test_dataloader)

    def get_label_dict(self, filename):
        """ Load class => name JSON file

        Returns:
            (dict): Dictionary of classes and names
        """
        with open(filename, 'r') as f:
            cat_to_name = json.load(f)

        return cat_to_name

    def process_image(self, filename):
        """Process a PIL image for use in a PyTorch model

        Args:
            filename (str): Filename to process

        Returns:
            numpy.array: Numpy array
        """

        image = Image.open(filename)
        return self.test_transform(image)
