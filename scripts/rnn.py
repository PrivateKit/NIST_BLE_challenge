import argparse
from sklearn.model_selection import train_test_split
import os
import random
import math
import time
import torch
import torch.nn as nn
from scripts.load_dataset import load_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', type=str, required=True)
    parser.add_argument('--seed', '-s', type=int, default=100, required=False)

    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    seed = args.seed

    random.seed(seed)

    print("The code for the RNN is currently on Google Colab: https://colab.research.google.com/drive/1EiCx_vuLSqIaIkvyploM6DCx5UzzNyxH?usp=sharing")

