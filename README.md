# Udacity Project - Image Classifier

My submission for the Udacity AI Programming with Python Nanodegree, Image Classifier project.


### Installation

- Requires Anaconda, Python 3.6, PyTorch
- `conda create -n py36 python=3.6` - Create environment with python 3.6
- `conda activate py36` - Activate environment
- `conda install pytorch torchvision python-resize-image cudatoolkit=10.2 -c pytorch` - [Install PyTorch](https://pytorch.org/get-started/locally/)

## Download Data

Download flower image data from [this link](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz) and place `train`, `test` and `valid` folders inside `data` folder.

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
$ python train.py data --checkpoint=vgg11 --arch=vgg11
Command Line Arguments:
data_dir = data
checkpoint = vgg11
checkpoint_dir = checkpoints
arch = vgg11
learning_rate = 0.001
hidden_units_1 = 2448
hidden_units_2 = 1224
dropout_1 = 0.2
dropout_2 = 0.2
gpu = True
labels_file = cat_to_name.json
epochs = 20

Checkpoint 'checkpoints/vgg11.pth' does not exist

Model training in session...

Epoch:   1/ 20  | Train Loss:  259.84157  | Validation Loss:   12.79943  | Validation Acc: 73.159%  | Duration:     58.785
Epoch:   2/ 20  | Train Loss:  123.87664  | Validation Loss:    8.00775  | Validation Acc: 83.817%  | Duration:     57.529
Epoch:   3/ 20  | Train Loss:   97.99223  | Validation Loss:    7.06926  | Validation Acc: 85.394%  | Duration:     57.276
Epoch:   4/ 20  | Train Loss:   87.35588  | Validation Loss:    6.24106  | Validation Acc: 87.091%  | Duration:     57.709
Epoch:   5/ 20  | Train Loss:   77.90076  | Validation Loss:    6.86781  | Validation Acc: 85.942%  | Duration:     57.164
Epoch:   6/ 20  | Train Loss:   76.74109  | Validation Loss:    5.38921  | Validation Acc: 89.582%  | Duration:     57.997
Epoch:   7/ 20  | Train Loss:   71.08278  | Validation Loss:    5.42153  | Validation Acc: 89.255%  | Duration:     57.273
Epoch:   8/ 20  | Train Loss:   67.55841  | Validation Loss:    5.91855  | Validation Acc: 88.846%  | Duration:     57.163
Epoch:   9/ 20  | Train Loss:   64.63595  | Validation Loss:    5.74725  | Validation Acc: 89.803%  | Duration:     57.067
Epoch:  10/ 20  | Train Loss:   63.89030  | Validation Loss:    6.41510  | Validation Acc: 88.620%  | Duration:     58.350
Epoch:  11/ 20  | Train Loss:   62.23059  | Validation Loss:    5.41267  | Validation Acc: 90.269%  | Duration:     58.035
Epoch:  12/ 20  | Train Loss:   60.43758  | Validation Loss:    5.11226  | Validation Acc: 91.264%  | Duration:     58.127
Epoch:  13/ 20  | Train Loss:   56.29512  | Validation Loss:    5.56420  | Validation Acc: 89.428%  | Duration:     58.220
Epoch:  14/ 20  | Train Loss:   57.98925  | Validation Loss:    5.48434  | Validation Acc: 90.149%  | Duration:     58.282
Epoch:  15/ 20  | Train Loss:   58.99926  | Validation Loss:    5.93572  | Validation Acc: 90.558%  | Duration:     58.019
Epoch:  16/ 20  | Train Loss:   55.37423  | Validation Loss:    5.62315  | Validation Acc: 90.817%  | Duration:     58.520
Epoch:  17/ 20  | Train Loss:   56.75518  | Validation Loss:    5.85219  | Validation Acc: 90.644%  | Duration:     59.248
Epoch:  18/ 20  | Train Loss:   57.25512  | Validation Loss:    6.12152  | Validation Acc: 91.433%  | Duration:     58.359
Epoch:  19/ 20  | Train Loss:   54.85106  | Validation Loss:    5.46395  | Validation Acc: 90.904%  | Duration:     59.307
Epoch:  20/ 20  | Train Loss:   55.29797  | Validation Loss:    5.93761  | Validation Acc: 91.418%  | Duration:     59.390

Testing model against test data...

Test Loss:    4.84023  | Test Acc: 90.168%  | Duration:      6.060s

Total Train Duration: 1399.348s
```

```sh
$ python predict.py assets/spring_crocus.jpg vgg11
Command Line Arguments:
image = assets/spring_crocus.jpg
checkpoint = vgg11
checkpoint_dir = checkpoints
arch = vgg16
gpu = True
category_names = cat_to_name.json
top_k = 10

Loading checkpoint checkpoints/vgg11.pth

75.295% spring crocus [67]
20.951% siam tulip [39]
1.251% columbine [84]
1.235% cyclamen [88]
0.545% sweet pea [4]
0.248% canterbury bells [3]
0.179% garden phlox [32]
0.118% monkshood [9]
0.048% stemless gentian [28]
0.042% grape hyacinth [25]
```