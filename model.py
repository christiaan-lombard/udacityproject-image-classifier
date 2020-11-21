#!/usr/bin/env python3
#
# PROGRAMMER: Christiaan Lombard
# DATE CREATED: 2020-11-21
# REVISED DATE: 2020-11-21
# PURPOSE: Define model architecture
#
#

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models


# available models
models = {
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
}


class GPUUnvailableError(Exception):
    """Error to indicate GPU is unavailable on host machine"""
    pass


class Model():
    """Wrapper for a torch nn.Module with support for dynamic attributes
    """

    def __init__(self, class_to_idx,
                 arch='vgg16', out_features=102,
                 hidden_units_1=2448, hidden_units_2=1224,
                 dropout_1=0.2, dropout_2=0.2, use_gpu=True):
        """Create instance of model

        Args:
            class_to_idx (dict): Class to idx dictionary as provided by train dataloader
            arch (str, optional): The existing model architecture to use as feature extractor. Defaults to 'vgg16'.
            out_features (int, optional) Number of output features. Defaults to 102.
            hidden_units_1 (int, optional): Number of units in 1st hidden layer. Defaults to 2448.
            hidden_units_2 (int, optional): Number of units in 2nd hidder layer. Defaults to 1224.
            dropout_1 (float, optional): Dropout rate for 1st hidden layer. Defaults to 0.2.
            dropout_2 (float, optional): Dropout rate for 2nd hidden layer. Defaults to 0.2.
            use_gpu (bool, optional): Whether model should use GPU. Defaults to True.

        Raises:
            ValueError: If arch does not exists
            GPUUnvailableError: If GPU selected but not available
        """

        self.arch = arch
        self.class_to_idx = class_to_idx
        self.out_features = out_features
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2

        # load pretrained model
        if arch in models:
            self.model = models[arch](pretrained=True)
        else:
            raise ValueError("Unknown model arch '%s'" % (arch))

        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False

        # mirror model classifier in_features
        in_features = self.model.classifier[0].in_features

        # define classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units_1),
            nn.ReLU(),
            nn.Dropout(p=dropout_1),
            nn.Linear(hidden_units_1, hidden_units_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_2),
            nn.Linear(hidden_units_2, out_features),
            nn.LogSoftmax(dim=1)
        )

        # replace model's classifier with our own
        self.model.classifier = self.classifier

        # switch to gpu if selected
        if use_gpu:
            if not torch.cuda.is_available():
                raise GPUUnvailableError(
                    "CUDA capable GPU is not available, use CPU instead.")
            self.device = torch.device('cuda:0')
            self.model.to(self.device)
        else:
            self.device = torch.device('cpu')

    def forward(self, x):
        """Feedforward to model

        Args:
            x (Tensor): Input image tensor

        Returns:
            Tensor: Predictions (LogSoftmax)
        """

        return self.model.forward(x)

    def train(self):
        """Training Mode"""
        self.model.train()

    def eval(self):
        """Eval Mode"""
        self.model.eval()

    def predict(self, image, topk=5):
        """Predict images classes and return top predictions

        Args:
            image (Tensor): Image tensor as proccessed by ImageDataLoader
            topk (int, optional): Number of top classes to return. Defaults to 5.

        Returns:
            (list, list): List of propabilities and list of classes
        """
        idx_to_class = dict(
            zip(self.class_to_idx.values(), self.class_to_idx.keys()))

        with torch.no_grad():
            # TODO: Implement the code to predict the class from an image file

            self.eval()
            image = torch.Tensor(image.unsqueeze(0)).to(self.device)

            log_probs = self.forward(image).cpu()
            probs = torch.exp(log_probs)

            # top predictions
            top_probs, top_class = probs.topk(topk, dim=1)
            top_class = [idx_to_class[i.item()] for i in top_class.flatten()]
            top_probs = top_probs.flatten()

            return (top_probs.numpy().tolist(), top_class)
