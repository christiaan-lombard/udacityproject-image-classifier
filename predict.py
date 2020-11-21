import time
import argparse
from command_line import print_command_line_arguments, str2bool
from model import Model
from model_data_loader import ModelDataLoader
from model_trainer import ModelTrainer
from model_loader import ModelLoader

def get_input_args():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('image', type=str, help="Image filename to predict")
    parser.add_argument('checkpoint', type=str, default="checkpoint", help="Checkpoint name")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Checkpoint data folder to save/load checkpoint")
    parser.add_argument('--arch', type=str, default="vgg16", help="CNN Model Architecture")
    parser.add_argument('--gpu', type=str2bool, default=True, help="Model use GPU (requires CUDA support)")
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help="Labels json file to use")
    parser.add_argument('--top_k', type=int, default=10, help="Number of top classes to display")

    return parser.parse_args()


def main():
    # Measures total program runtime by collecting start time
    start_time = time.time()

    in_arg = get_input_args()

    # Function that checks command line arguments using in_arg
    print_command_line_arguments(in_arg)
    print()

    dataloader = ModelDataLoader('data')
    category_names = dataloader.get_label_dict(in_arg.category_names)
    loader = ModelLoader(in_arg.checkpoint_dir)

    if loader.checkpoint_exists(in_arg.checkpoint):
        print("Loading checkpoint %s"%(loader.get_checkpoint_path(in_arg.checkpoint)))
        model = loader.load_checkpoint(in_arg.checkpoint, model_use_gpu=in_arg.gpu, with_trainer=False)
    else:
        print("Checkpoint '%s' does not exist. Exiting."%(loader.get_checkpoint_path(in_arg.checkpoint)))
        return

    print()

    image = dataloader.process_image(in_arg.image)

    probs, classes = model.predict(image, in_arg.top_k)
    labels = [category_names[c] for c in classes]

    for prob, clas, label in zip(probs, classes, labels):
        print("%5.3f%% %s [%s]"%(prob*100, label, clas))



# Call to main function to run the program
if __name__ == "__main__":
    main()
