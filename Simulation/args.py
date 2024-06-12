import argparse

parser = argparse.ArgumentParser(description='Running arguments')

parser.add_argument('--circuit', type=str, default='', help='specify a circuit name, e.g. SingleStage')
parser.add_argument('--model', type=str, default='MLP', help='specify a model, e.g., MLP')
parser.add_argument('--npoints', type=int, default=1, help='number of points to simulate')

args = parser.parse_args()