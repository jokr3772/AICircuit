# circuitPilot
abstract #TO BE ADDED

## Table of Contents

  * [Requirements](#requirements)
  * [Usage](#usage)
    * [Simple Run](#simple-run)
    * [Description](#description)
    * [Results](#results)
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

Second, generate your own **train config** file, or use our default one. For using the default config file:

```
python3 main.py
```

As an example, if you want to address your own config path:

```
python3 main.py --path ./Config/train_config.yaml
```

### Description
Here is a description for each parameter in the `train_config.yaml` file:

  * `model_config`: a list of different model configs:
    * `model`: 
      * KNeighbors
      * RandomForest
      * SupportVector
      * MultiLayerPerceptron
      * Transformer
    * other model args: add desired model arguments to the `model_config` (no ordering). For now, you can add Transformer parameters: `dim_model`,
    `num_heads`, `num_encoder_layers`, `dim_hidden`, `dropout_p`. if no model args added, uses the default ones.
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
  * `loss_per_epoch`: for KNN, RF, or SVR is False, else can be True or False by config value. default is True for MLP and Transformer.
  * `compare_method`: if True, for each subset and circuit, plots different models' loss (MLP and Transformer) on the same figure. 
  * `kfold`: if True, K fold cross validation is performed. K is defined based on `subset` value. if False, no K fold cross validation.
  * `independent_kfold`: if True, train and test data is selected independently from previous folds, if false, dataset is divided to k parts at first and each time one part is selected for test and the remaining for train.
  * `save_format`: 
    * csv
    * numpy
    * anything else: no saving

### Results
 
In each run, find plots in the corresponding folder in `out_plot` folder and loss values in the corresponding folder in `out_result` folder. Also if you set the `train_config.yaml` in a way that it saves the predictions, you can find the prediction files in the mentioned directory at the circuit's yaml file.

## Visualization
 
#TO BE ADDED

## Documentation
  
  For more details, visit our paper [papertitle](link).

## Where to ask for help

#TO BE ADDED

## Contributors

#TO BE ADDED
