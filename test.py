
import torch
import torch.nn as nn
import yaml
from addict import Dict
import click
import os

@click.command()
@click.option('--device', default='0')
def main(device):
    print(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    print(torch.cuda.current_device())

if __name__ == "__main__":
    main()