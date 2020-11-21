import torch
from torch import nn, optim
import time
import numpy as np

class ModelTrainer():

    def __init__(self, model, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.model = model
        self.trained_epochs = 0

        # define loss function
        self.criterion = nn.NLLLoss()

        # define optimizer
        self.optimizer = optim.Adam(
                self.model.classifier.parameters(),
                lr=self.learning_rate
            )


    def train_epochs(self, num_epochs, train_dataloader, valid_dataloader):


        for e in range(num_epochs):
            train_loss = 0
            validation_loss = 0
            accuracies = []
            start = time.time()

            # iterate training batches
            for images, labels in train_dataloader:
                train_loss += self.train(images, labels)
                pass

            # iterate validation batches
            with torch.no_grad():
                for images, labels in valid_dataloader:
                    loss, accuracy = self.validate(images, labels)
                    validation_loss += loss
                    accuracies.append(accuracy)
                    pass

            # increment total trained epochs
            self.trained_epochs += 1
            yield {
                'epoch': e,
                'train_loss': train_loss,
                'validation_loss': validation_loss,
                'validation_accuracy': np.mean(accuracies),
                'duration': time.time() - start,
            }

    def test(self, test_dataloader):

        test_loss = 0
        accuracies = []
        start = time.time()

        # iterate test batches
        with torch.no_grad():
            for images, labels in test_dataloader:
                loss, accuracy = self.validate(images, labels)
                test_loss += loss
                accuracies.append(accuracy)
                pass

        return {
            'test_loss': test_loss,
            'test_accuracy': np.mean(accuracies),
            'duration': time.time() - start
        }


    def train(self, images, labels):

        # training mode
        self.model.train()

        # move data to appropriate device
        images = images.to(self.model.device)
        labels = labels.to(self.model.device)

        # reset gradients
        self.optimizer.zero_grad()

        # predict, calc loss, backprop, learn
        outputs = self.model.forward(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, images, labels):

        # eval mode
        self.model.eval()

        # move data to appropriate device
        images = images.to(self.model.device)
        labels = labels.to(self.model.device)

        # predict, calc loss
        log_ps = self.model.forward(images)
        loss = self.criterion(log_ps, labels)
        ps = torch.exp(log_ps)

        # calc accuracy of top predictions
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))

        return (loss.item(), accuracy.item())

    def get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']