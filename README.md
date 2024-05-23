# circuitPilot
abstract #TO BE ADDED

## Table of Content

  * [Requirements](#requirements)
  * [Usage](#usage)
    * [Simple Run](#simple-run)
    * [Description](#description)
  * [Contributing](#contributing)
  * [Support and Migration](#support-and-migration)
  * [License](#license)

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
