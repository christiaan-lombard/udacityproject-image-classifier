#!/usr/bin/env python3
#
# PROGRAMMER: Christiaan Lombard
# DATE CREATED: 2020-11-21
# REVISED DATE: 2020-11-21
# PURPOSE: Define model checkpoint loader
#
#


from model import Model
import torch
from os import path
from model_trainer import ModelTrainer


class ModelLoader:
    """Loads and saves model checkpoints
    """

    def __init__(self, directory):
        """Create instance of model loader

        Args:
            directory (str): Directory of checkpoints
        """
        self.directory = directory

    def save_checkpoint(self, name, model, trainer):
        """Save model checkpoint

        Args:
            name (str): Name of checkpoint
            model (Model): Model to save
            trainer (ModelTrainer): Model trainer (to extract optimizer parameters)
        """

        checkpoint = {
            'name': name,
            'arch': model.arch,
            'out_features': model.out_features,
            'hidden_units_1': model.hidden_units_1,
            'hidden_units_2': model.hidden_units_2,
            'dropout_1': model.dropout_1,
            'dropout_2': model.dropout_2,
            'model_state': model.model.state_dict(),
            'class_to_idx': model.class_to_idx,
            'optimizer_state': trainer.optimizer.state_dict(),
            'trained_epochs': trainer.trained_epochs,
            'learning_rate': trainer.learning_rate,
        }
        torch.save(checkpoint, self.get_checkpoint_path(name))

    def load_checkpoint(self, name, with_trainer=True, model_use_gpu=True):
        """Load model checkpoint and setup the Model and ModelTrainer

        Args:
            name (str): Checkpoint name
            with_trainer (bool, optional): Whether to load the ModelTrainer. Defaults to True.
            model_use_gpu (bool, optional): Setup the Model with GPU enabled. Defaults to True.

        Returns:
            (Model, ModelTrainer): Returns model and trainer
        """
        checkpoint = torch.load(self.get_checkpoint_path(name))

        # recover model state
        model = Model(
            checkpoint['class_to_idx'],
            arch=checkpoint['arch'],
            out_features=checkpoint['out_features'],
            hidden_units_1=checkpoint['hidden_units_1'],
            hidden_units_2=checkpoint['hidden_units_2'],
            dropout_1=checkpoint['dropout_1'],
            dropout_2=checkpoint['dropout_2'],
            use_gpu=model_use_gpu
        )

        model.model.load_state_dict(checkpoint['model_state'])
        model.model.class_to_idx = checkpoint['class_to_idx']

        # recover trainer state
        if with_trainer:
            trainer = ModelTrainer(model, checkpoint['learning_rate'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            trainer.trained_epochs = checkpoint['trained_epochs']
            return (model, trainer)
        else:
            return model

    def get_checkpoint_path(self, name):
        """Get the filepath of a given checkpoint name

        Args:
            name (str): Checkpoint name

        Returns:
            str: Path
        """
        return self.directory + '/' + name + '.pth'

    def checkpoint_exists(self, name):
        """Check if a checkpoint file exists

        Args:
            name (str): Checkpoint name

        Returns:
            bool: Whether the checkpoint file exists
        """
        return path.exists(self.get_checkpoint_path(name))
