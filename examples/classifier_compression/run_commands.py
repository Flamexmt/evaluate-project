import os


def run():
    os.system(
        "python compress_classifier.py -a= wideresnet --data ../../data.cifar --workers 3 --batch-size 1024 - --evaluate --confusion --adv 1 --resume-from ../../outputsdata/wideresnet/checkpoint.pth.tar --out-dir ../../outputsdata/eval/ --quantized 8 --cpu")



import datetime

run()
