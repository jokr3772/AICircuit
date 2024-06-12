# ml-circuit
Benchmark for analog circuit design with machine learning.

---

## Overview

In this project, we test several machine learning based methods on analog circuit design.
In a high level, when using machine learning to design an analog circuit, a model receives several performance metrics (e.g., bandwidth, power consumption) as input,
then the model outputs design parameters. We test the following methods:

- standard MLP model
- RL with GCN model

## Circuit and Dataset Collection

The following circuits are tested:
- Single Stage Amplifier
- Cascode Amplifier
- Two Stage Amplifier
- Voltage Control Oscillator (VCO)
- Power Amplifier (PA)
- Mixer

## Models

## Run

```bash
# run simulation on the Power Amplifier (PA)

python main.py --circuit=PA
```