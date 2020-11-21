import time
import argparse
from command_line import print_command_line_arguments, str2bool
from model import Model
from model_data_loader import ModelDataLoader
from model_trainer import ModelTrainer
from model_loader import ModelLoader

def get_input_args():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('data_dir', type=str, default="data", help="Image data folder")
    parser.add_argument('--checkpoint', type=str, default="checkpoint", help="Checkpoint name")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Checkpoint data folder to save/load checkpoint")
    parser.add_argument('--arch', type=str, default="vgg16", help="CNN Model Architecture")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--hidden_units_1', type=int, default=2448, help="Hidden layer 1 neurons")
    parser.add_argument('--hidden_units_2', type=int, default=1224, help="Hidden layer 2 neurons")
    parser.add_argument('--dropout_1', type=float, default=0.2, help="Hidden layer 1 dropout rate")
    parser.add_argument('--dropout_2', type=float, default=0.2, help="Hidden layer 2 dropout rate")
    parser.add_argument('--gpu', type=str2bool, default=True, help="Model use GPU (requires CUDA support)")
    parser.add_argument('--labels_file', type=str, default='cat_to_name.json', help="Labels json file to use")
    parser.add_argument('--epochs', type=int, default=20, help="Epochs to train")

    return parser.parse_args()


def main():
    # Measures total program runtime by collecting start time
    start_time = time.time()

    in_arg = get_input_args()

    # Function that checks command line arguments using in_arg
    print_command_line_arguments(in_arg)
    print()

    dataloader = ModelDataLoader(in_arg.data_dir)
    labels = dataloader.get_label_dict(in_arg.labels_file)
    train_dataset, train_dataloader = dataloader.get_train_data()
    valid_dataset, valid_dataloader = dataloader.get_validation_data()
    test_dataset, test_dataloader = dataloader.get_test_data()

    loader = ModelLoader(in_arg.checkpoint_dir)

    if loader.checkpoint_exists(in_arg.checkpoint):
        print("Loading checkpoint %s"%(loader.get_checkpoint_path(in_arg.checkpoint)))
        model, trainer = loader.load_checkpoint(in_arg.checkpoint, model_use_gpu=in_arg.gpu)
        print("Epochs trained so far: %d"%(trainer.trained_epochs))
    else:
        print("Checkpoint '%s' does not exist"%(loader.get_checkpoint_path(in_arg.checkpoint)))
        model = Model(train_dataset.class_to_idx,
                        arch=in_arg.arch, use_gpu=in_arg.gpu,
                        hidden_units_1=in_arg.hidden_units_1, hidden_units_2=in_arg.hidden_units_2,
                        dropout_1=in_arg.dropout_1, dropout_2=in_arg.dropout_2)
        trainer = ModelTrainer(model, learning_rate=in_arg.learning_rate)


    print()
    print("Model training in session...")
    print()

    epochs = in_arg.epochs

    for result in trainer.train_epochs(epochs, train_dataloader, valid_dataloader):
        print(
            "Epoch: %3d/%3d"%(result['epoch']+1, epochs),
            " | Train Loss: %10.5f"%(result['train_loss']),
            " | Validation Loss: %10.5f"%(result['validation_loss']),
            " | Validation Acc: %6.3f%%"%(result['validation_accuracy'] * 100),
            " | Duration: %10.3fs"%(result['duration'])
        )

    print()
    print("Testing model against test data...")
    print()

    test_result = trainer.test(test_dataloader)

    print(
        "Test Loss: %10.5f"%(test_result['test_loss']),
        " | Test Acc: %6.3f%%"%(test_result['test_accuracy'] * 100),
        " | Duration: %10.3fs"%(test_result['duration'])
    )

    loader.save_checkpoint(in_arg.checkpoint, model, trainer)

    print()
    print("Total Train Duration: %.3fs"%(time.time() - start_time))





# Call to main function to run the program
if __name__ == "__main__":
    main()
