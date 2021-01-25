import os


def run():
    os.system("python compress_classifier.py -a=resnet44_cifar --data ../../data.cifar --workers 8 --batch-size 2000 --gpus 0 --resume-from ../../outputsdata/resnet44/checkpoint.pth.tar --out-dir ../../outputsdata/new_eval_result --evaluate --adv 1 --cw-attack 1 --pgd-attack 1 --square-attack 1 --adv-batch-size 2000")
    os.system("python compress_classifier.py -a=vgg16_cifar --data ../../data.cifar --workers 8 --batch-size 2000 --gpus 0 --resume-from ../../outputsdata/vgg16_cifar/checkpoint.pth.tar --out-dir ../../outputsdata/new_eval_result --evaluate --adv 1 --cw-attack 1 --pgd-attack 1 --square-attack 1 --adv-batch-size 2000")
    os.system("python compress_classifier.py -a=wide_resnet50_2_cifar --data ../../data.cifar --workers 8 --batch-size 2000 --gpus 0 --resume-from ../../outputsdata/wideresnet_cifar/checkpoint.pth.tar --out-dir ../../outputsdata/new_eval_result --evaluate --adv 1 --cw-attack 1 --pgd-attack 1 --square-attack 1 --adv-batch-size 2000")



import datetime

run()
