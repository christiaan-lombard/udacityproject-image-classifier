# Udacity Project - Image Classifier

My submission for the Udacity AI Programming with Python Nanodegree, Image Classifier project.


### Installation

- Requires Anaconda, Python 3.6, PyTorch
- `conda create -n py36 python=3.6` - Create environment with python 3.6
- `conda activate py36` - Activate environment
- `conda install pytorch torchvision python-resize-image cudatoolkit=10.2 -c pytorch` - [Install PyTorch](https://pytorch.org/get-started/locally/)


### Scripts

 - `python train.py data_dir` - Train model on images in data directory
   - Prints out training loss, validation loss, and validation accuracy as the network trains
   - Options:
     - Set directory to save checkpoints: `python train.py data --checkpoint_dir checkpoints`
     - Choose architecture: `python train.py data --arch "vgg13" --checkpoint=vgg13`
     - Set hyperparameters: `python train.py data --learning_rate 0.01 --hidden_units_1 512 --epochs 20`
     - Use GPU for training: `python train.py data --gpu`
 - `pyton predict.py input_image checkpoint` - Run model prediction
   - Options:
     - Return top K most likely classes: `python predict.py assets/spring_crocus.jpg checkpoint --top_k 3`
     - Use a mapping of categories to real names: `python predict.py assets/spring_crocus.jpg checkpoint --category_names cat_to_name.json`
     - Use GPU for inference: `python predict.py assets/spring_crocus.jpg checkpoint --gpu`

### Example output:

```sh

```