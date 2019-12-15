import torch
from torchvision import datasets
import torchvision.transforms as transforms
import argparse
ALL_ASPECT_INCLUDED=('accuracy','fairness','robustness','generalizability')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate tool main')
    parser.add_argument('--data',  help='path to dataset')
    parser.add_argument('--pretrained',help='path to pretrained model parameter')
    parser.add_argument('--aspect', type=lambda s: s.lower(),default='accuracy',choices=ALL_ASPECT_INCLUDED,help='the aspect to be evaluated'.join(ALL_ASPECT_INCLUDED))
    parser.add_argument('--log',help='path to save log')