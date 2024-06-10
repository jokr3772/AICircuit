# AICircuit: A Multi-Level Dataset and Benchmark for AI-Driven Analog Integrated Circuit Design (NeurIPS 2024)

## Overview

AICircuit is a comprehensive multi-level dataset and benchmark for developing and evaluating ML algorithms in analog and radio-frequency circuit design. AICircuit comprises seven commonly used basic circuits and two complex wireless transceiver systems composed of multiple circuit blocks, encompassing a wide array of design scenarios encountered in real-world applications. We extensively evaluate various ML algorithms on the dataset, revealing the potential of ML algorithms in learning the mapping from the design specifications to the desired circuit parameters. 

## Table of Contents

  * [Requirements](#requirements)
  * [Usage](#usage)
    * [Simple Run](#simple-run)
    * [Description](#description)
    * [Results](#results)
  * [Visualization](#visualization)
  * [Contributors](#contributors)
  <!-- * [Documentation](#documentation)
  * [Where to ask for help](#where-to-ask-for-help) -->

## Requirements

To install the requirements you can use:

```
pip install -r requirements.txt
```

## Usage

### Simple Run
To run the code, first change the path for your input data and output results in **arguments** of config files in `Config/Circuits` based on your needs. <br>

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
    * other model args: add desired model arguments to the `model_config` (no ordering). For now, you can add:
      * KNeighbors parameters: `n_neighbors`, `weights`
      * RandomForest parameters: `n_estimators`, `criterion`
      * Transformer parameters: `dim_model`, `num_heads`, `num_encoder_layers`, `dim_hidden`, `dropout_p`. 
    if no model args added, uses the default ones.
  * `subset`: a list of the fraction of data used for training.
  * `circuits`: a list of circuits from this list:
    * CSVA: Common-Source Voltage Amplifier (CSVA)
    * CVA: Cascode Amplifier
    * TSVA: Two-Stage Voltage Amplifier
    * LNA: Low-Noise Amplifier
    * Mixer
    * VCO: Voltage-Controlled Oscillator
    * PA: Power Amplifier
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

<!-- ## Documentation
  
  For more details, visit our paper [here](link).

## Where to ask for help

If you have any questions, feel free to open a [discussion](https://github.com/AsalMehradfar/AICircuit/discussions) and ask your question. -->

## Contributors

[Asal Mehradfar](https://github.com/AsalMehradfar), [Xuzhe Zhao](https://github.com/XuzheZ827), [Yue Niu](https://github.com/yuehniu), [Sara Babakniya](https://github.com/SaraBabakN), [Mahdi Alesheikh](https://github.com/malshei), [Hamidreza Aghasi](https://hie.eng.uci.edu/), [Salman Avestimehr](https://www.avestimehr.com/)
