

This project is used to compress and evaluate cnn models. Compression part is from https://github.com/NervanaSystems/distiller/. 

# Prerequisite
## After you have a python envrioment (>=3.7) and pip, run the following command.
$ cd distiller
$ pip3 install -e .

# Run
## Run the test
$ cd distiller/tests
$ pytest

## Run the training(eg. resnet44 training with no compression, the command can find in command.txt)
$ cd examples/classifier_compression
$ python compress_classifier.py -a resnet44_cifar --data ../../data.cifar --lr 0.005 -p 50 --epochs 100 --gpus 0 --out-dir ../../outputsdata/ --batch-size 64 --workers 3 --confusion

## Run the evaluation
$ cd examples/classifier_compression
$ python compress_classifier.py -a=resnet44_cifar --data ../../data.cifar --workers 3 --batch-size 64 --evaluate --confusion --adv 1 --resume-from ../../outputsdata/resnet44/checkpoint.pth.tar --out-dir ../../outputsdata/eval/


