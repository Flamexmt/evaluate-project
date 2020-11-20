import os


def run(arch, filepath):
    os.system(
        "python compress_classifier.py -a=" + arch + " --data ../../data.cifar --workers 3 --batch-size 128 --evaluate --confusion --adv 1 --resume-from ../../outputsdata/" + filepath + "/checkpoint.pth.tar --out-dir ../../outputsdata/eval/cw_inf --gpus 0")


run_list = [('resnet20_cifar', 'resnet20'),('resnet20_cifar', 'resnet20-apgpruning'), ('simplenet_cifar ', 'resnet20-realkd'),
            ('simplenet_cifar', 'resnet20-realkd-apgpruning'), ('resnet44_cifar', 'resnet44'),
            ('resnet44_cifar', 'resnet44-apgpruning'), ('resnet20_cifar', 'resnet44-kd'),
            ('resnet20_cifar', 'resnet44-kd-apgpruning'),('vgg16_cifar', 'vgg16-base(50)'), ('vgg16_cifar', 'vgg16-base(50)-pruning'),
            ('vgg11_cifar ', 'vgg16-kd-vgg11(50-100)'),
            ('vgg11_cifar', 'vgg16-kd-vgg11(50-100)-pruning(100-180)'),
            ('wideresnet_cifar', 'wideresnet'),
            ('wideresnet_cifar', 'wideresnet-pruning'),
            ('resnet44_cifar', 'wideresnet-kd-resnet44'),
            ('resnet44_cifar', 'wideresnet-kd-resnet44-pruning'),('resnet32_cifar', 'resnet44-kd-resnet32'), ('resnet32_cifar', 'resnet44-kd-resnet32-pruning'),
            ('resnet26_cifar', 'resnet44-kd-resnet26'), ('resnet32_cifar', 'resnet44-kd-resnet26-pruning'),]
import datetime
for item in run_list:
    start_time = datetime.datetime.now()
    print('now runing', item, 'at', start_time)
    run(item[0], item[1])
    end_time = datetime.datetime.now()
    print('finish runing', item, 'at', end_time)
    print((end_time - start_time))
    break
print('finish')