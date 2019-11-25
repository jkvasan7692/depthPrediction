# Depth Prediction Deep Learning Model

Description to be added here

## Getting Started

### Prerequisites

The code is implemented in Python and has the following Dependency :
1. Python3
2. Pytorch
3. Cuda 10.0
4. CuDNN

### Installing

Description to be added

### Compilation

Description to be added

### Directory and Files
The project directory structure is as follows:
1. net - Contains the python files for the depth prediction model
  * classifier.py : The python file containing the network model.
     Loads the pretrained ResNet-50 layer network with pretrained weights, contains simple upconvolution layers
2. torchlight
  * io.py - function to print log onto console, save features(commented )
  * gpu.py - function ngpu to count number of GPUs

3. utils - Utilities to load dataset and processor providing the supporting helper functions(train, test, print)
  * processor.py - Contains functions for training, testing, evaluation.
                   Invokes the dataset loader class, depth prediction network
                   Handles the command-line arguments, invokes torchlight class functionality to print, save the features
  * loader.py - Loads the dataset for training, testing, evaluation

4. model_output - Stores the model outputs including checkpoint results - To be implemented

## ToDo list
1. Model - Implement the weight initialization for the other layers - [done], Modify the up-sampling instead of using UpsampleNearest2d function - [Kirthi]  - will do today
2. Model - Need to check for the up-projection layer and fast up-convolution and up-projection layers
3. Loader - Class to be modified to load the dataset for the current application. Need to integrate this - [done]
   with the rest of the code in the main1.py and processor1.py file [done]
4. Processor - Saving of the best features to be completed - Least priority
5. Processor - Check for the TBD tagged comments and resolve them based on priority - [Noted all points]
6. Processor - Implement saving of the intermediate checkpoints for debugging during training. - [Done. Need to save epochs based on some benchmark]
7. Processor - Unit test the functionalities - [Kithi] - [done]
8. Processor - Adjusting learning rate based on change between two mean loss steps
9. Loader - Have a train, test and evaluation dataset.

## Authors

**Janakiraman Kirthivasan** - *Initial work* - [jkvasan7692](https://github.com/jkvasan7692)
**Nantha Kumar** - *Initial work* - [nantha007](https://github.com/nantha007)

## Results link
Description to be added
