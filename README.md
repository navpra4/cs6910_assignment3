# cs6910_assignment3

Assignment 2 on Recurrent Neural Network

## Introduction
The goal of this assignment is twofold: (i) train a RNN model from scratch and learn how to tune the hyperparameters and translate english words to local language (ii) Use Attentional networks for the same

## Requirements
This project requires following 

- Python
- WandB
- Numpy
- PyTorch
- matplotlib


## Understanding the datasets used

The dataset is a large-scale dataset containing english and local language words. 



## Installation

1. Clone the repository:

   bash
   git clone 
   

2. Install dependencies:

   bash
   pip install -r requirements.txt
   



## EXPLAINATION OF PY FILES

### train_model.py
This file contains our model that is trained from scratch and tested on the testing data avalable at test_path.


### Load_data.py
This file contains functions for loading datasets into tensors.

## encoder.py 
Contains the encoder class

## decoder.py

contains the decoder class and attndecoder class

## evalutate.py

Contains the functions need To find the accuracy and cost of the model.


## Usage

 Train the RNN:

   Run the following command to train the RNN:

   bash
   python train_model.py <list of arguments passed>

   
   You can pass the hyperparameters in command line before training. train_path and test_path are required parameters so we have to pass it to train the model for rest hyperparameter it will take default value.

## INSTRUCTIONS ON HOW TO RUN 

* Create a wandb Account before running train.py.
* Give the api key to your account when prompted.
* install packages as mentioned in the Installation section
  
The following table contains the arguments supported by the train.py file
|Name|	Default Value|	Description|
|:----:| :---: |:---:|
|-wp, --wandb_project	|myprojectname	|Project name used to track experiments in Weights & Biases dashboard|
|-we, --wandb_entity|	myname	|Wandb Entity used to track experiments in the Weights & Biases dashboard.|
|-trp, --train_path|		|Path to training dataset|
|-tsp, --test_path|		|Path to testing dataset|
|-e, --epochs	|5	|Number of epochs to train network.|
|-b, --batch_size|	64	|Batch size used to train  network.|
|-lr, --learning_rate	|0.0001	|Learning rate used to optimize model parameters|
|-dpo, --dropout | 0.2 | choices: int|
|-bd,--bidirectional | True| choices:[False, True]|
|-atn,--attention|True| choices: [False, True]|
|-d,--dropout|0.3 | dropout value|
|-emsz,--embedding_size| 128 | embedding size  in the network|
|-sz,--hidden_size| 5 | size of Hidden layer|



The following table contains the arguments supported by the pre_train.py file rest are the default values.
|Name|	Default Value|	Description|
|:----:| :---: |:---:|
|-wp, --wandb_project	|myprojectname	|Project name used to track experiments in Weights & Biases dashboard|
|-we, --wandb_entity|	myname	|Wandb Entity used to track experiments in the Weights & Biases dashboard.|
|-trp, --train_path|		|Path to training dataset|
|-tsp, --test_path|		|Path to testing dataset|
