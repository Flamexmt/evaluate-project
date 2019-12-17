import torch
from torchvision import datasets
import torchvision.transforms as transforms
import argparse
import accuracy
ALL_ASPECT_INCLUDED = ('all', 'accuracy', 'fairness', 'robustness', 'generalizability')
DATASET=('mnist','cifar10','imagent')
MODEL_ARCH=('simple_mnist','plain20_cifar','preresnet','resnet20_cifar','resnet32_cifar','resnet44_cifar','resnet56_cifar','simplenet_cifar','vgg')#还有一些模型没加上去
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate tool main')
    parser.add_argument('--dataset_path', help='path to dataset')
    parser.add_argument('--dataset',help='dataset in mnist,cifar10 and imagenet',choices=DATASET)
    parser.add_argument('--model',help='the model architecture',choices=MODEL_ARCH)
    parser.add_argument('--checkpoint_path', help='path to pretrained model parameter')
    parser.add_argument('--aspect', type=lambda s: s.lower(), default='accuracy', choices=ALL_ASPECT_INCLUDED,
                        help='the aspect to be evaluated'.join(ALL_ASPECT_INCLUDED))
    parser.add_argument('--batch_size',help='batch_size',type=int,default=16)
    parser.add_argument('--log', help='path to save log')
    args = parser.parse_args()
    if args.aspect == 'accuracy':
        accuracy.test_accuracy(args)
