# circuitPilot
abstract #TO BE ADDED

## Table of Content

  * [Requirements](#requirements)
  * [Usage](#usage)
    * [Simple Run](#simple-run)
    * [Description](#description)
  * [Visualization](#visualization)
  * [Documentation](#documentation)
  * [Where to ask for help](#where-to-ask-for-help)
  * [Contributors](#contributors)

## Requirements

To install the requirements you can use:

```
pip install -r requirements.txt
```

## Usage

### Simple Run
To run the code, first change the path for your input data and output results in **arguments** of config files in `./Config/Circuits` based on your needs. <br>
If you want to keep the paths as they are, use `setup.py` to arrange the initial folders.

```
python3 Utils/setup.py
```

Second, generate your own **train config** file, or use our default one. For using the default config file:

```
python3 main.py
```

If you want to address your own config path (for example):

```
python3 main.py --path ./Config/train_config.yaml
```

### Description
Here is a description for each parameter in the `train_config.yaml` file:

  * `model_config`: a list of different model configs:
    * model: KNeighbors, RandomForest, SupportVector, MultiLayerPerceptron, Transformer
  * `subset`: a list of the fraction of data used for training.
  * `circuits`: a list of circuits from this list:
    * SingleStageAmplifier
    * TwoStageAmplifier
    * VCO (Voltage Control Oscillator)
    * PA (Power Amplifier)
    * Cascode
    * Mixer
    * Transmitter
    * Receiver
  * `epochs`: used for MLP and Transformer model, default is 100.
  * `loss_per_epoch`: if KNN, RF, or SVR is False, else can be True or False by config value. default is True for MLP and Transformer.
  * `compare_method`:
  * `kfold`: if True, k fold cross validation is performed. k is defined based on *subset* value. if False, no k fold cross validation.
  * `independent_kfold`: if True, train and test data is selected independetly from previous folds, if false, dataset is divided to k parts at first and each time one part is selected for test and th remaining for train.
  * `save_format`: 
    * csv
    * numpy
    * anything else: no saving

## Visualization


## Documentation
  
  For more details, visit our paper [papertitle](link).

## Where to ask for help

## Contributors