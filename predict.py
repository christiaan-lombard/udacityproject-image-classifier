import time
import argparse
from command_line import print_command_line_arguments
from model import Model
from model_data_loader import ModelDataLoader
from model_trainer import ModelTrainer
from model_loader import ModelLoader

def get_input_args():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('data_dir', type=str, default="data", help="Image data folder")
    parser.add_argument('checkpoint', type=str, default="checkpoint", help="Checkpoint name")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Checkpoint data folder to save/load checkpoint")
    parser.add_argument('--gpu', type=bool, default=True, help="Model use GPU (requires CUDA support)")
    parser.add_argument('--labels_file', type=str, default='cat_to_name.json', help="Labels json file to use")

    return parser.parse_args()


def main():
    # Measures total program runtime by collecting start time
    start_time = time.time()

    in_arg = get_input_args()

    # Function that checks command line arguments using in_arg
    print_command_line_arguments(in_arg)

    loader = ModelLoader(in_arg.checkpoint_dir)

    if loader.checkpoint_exists(in_arg.checkpoint):
        print("Loading checkpoint '%s'"%(loader.get_checkpoint_path(in_arg.checkpoint)))
        model = loader.load_checkpoint(in_arg.checkpoint, with_trainer=False)
    else:
        print("Checkpoint '%s' does not exist"%(loader.get_checkpoint_path(in_arg.checkpoint)))

    if(in_arg.gpu):
        model.use_gpu()

    dataloader = ModelDataLoader(in_arg.data_dir)
    labels = dataloader.get_label_dict(in_arg.labels_file)

    print()
    print("Model training in session...")
    print()

    epochs = in_arg.epochs

    for result in trainer.epochs(epochs, train_dataloader, valid_dataloader):
        print(
            "Epoch: %3d/%3d"%(result['epoch'], epochs),
            "Train Loss: %10.5f"%(result['train_loss']),
            "Validation Loss: %10.5f"%(result['validation_loss']),
            "Validation Accuracy: %6.3f%%"%(result['validation_accuracy'] * 100),
            "Duration: %10.3fs"%(result['duration'])
        )

    loader.save_checkpoint(in_arg.checkpoint, model, trainer)

    print()
    print("Total train duration: %.3fs"%(time.time() - start_time))





# Call to main function to run the program
if __name__ == "__main__":
    main()
