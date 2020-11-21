import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

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

    def __init__(self, class_to_idx,
        arch='vgg16', out_features=102,
        hidden_units_1=2448, hidden_units_2=1224,
        dropout_1=0.2, dropout_2=0.2, use_gpu=True):
        """Create a new model instance

        Args:
            labels (dict): Dictionary of labels, string keys (class idx), string values (class name)
            arch (str, optional): Model architecture. Defaults to 'vgg16'.

        """

        self.arch = arch
        self.class_to_idx = class_to_idx
        self.out_features = out_features
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2

        if arch in models:
            self.model = models[arch](pretrained=True)
        else:
            raise ValueError("Unknown model arch '%s'"%(arch))

        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False

        # mirror model classifier in_features
        in_features = self.model.classifier[0].in_features

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

        if use_gpu:
            if not torch.cuda.is_available():
                raise GPUUnvailableError("CUDA capable GPU is not available, use CPU instead.")
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
        self.model.train()

    def eval(self):
        self.model.eval()
