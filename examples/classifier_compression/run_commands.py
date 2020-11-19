import os


def run(arch, filepath):
    os.system(
        "python D:/study/model-compression/git-hub/evaluate-project/examples/classifier_compression/compress_classifier.py -a=" + arch + " --data ../../data.cifar --workers 3 --batch-size 128 --evaluate --confusion --adv 1 --resume-from ../../outputsdata/" + filepath + "/checkpoint.pth.tar --out-dir ../../outputsdata/eval/resnet20_44_on_cifar/")


run_list = [('resnet20_cifar', 'resnet20-apgpruning'), ('simplenet_cifar ', 'resnet20-realkd'),
            ('simplenet_cifar', 'resnet20-realkd-apgpruning'), ('resnet44_cifar', 'resnet44'),
            ('resnet44_cifar', 'resnet44-apgpruning'), ('resnet20_cifar', 'resnet44-kd'),
            ('resnet20_cifar', 'resnet44-kd-apgpruning')]
import datetime

for item in run_list:
    start_time = datetime.datetime.now()
    print('now runing', item, 'at', start_time)
    run(item[0], item[1])
    end_time = datetime.datetime.now()
    print('finish runing', item, 'at', end_time)
    print((end_time - start_time))
