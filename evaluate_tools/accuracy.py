import datetime
import os

import DataLoader
import modelFactory
import torch
import checkpointProcess
from torch.autograd import Variable
import torch.quantization as tq
import torch.nn as nn


def test_accuracy(args):
    if args.dataset == 'cifar10':
        print('cifar10')
        trainloader, test_loader = DataLoader.cifar10_loader(args.dataset_path, args.batch_size)
    elif args.dataset == 'minst':
        trainloader, test_loader = DataLoader.mnist_loader(args.dataset_path, args.batch_size)
    elif args.dataset == 'imagenet':
        print('imagenet')
    else:
        test_loader = None
    test(test_loader, args, )


def quntization(model):
    torch.backends.quantized.engine = 'qnnpack'
    print_size_of_model(model)
    model.fuse_model()
    qmodel=tq.quantize_dynamic(model,inplace=True)
    print_size_of_model(qmodel)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def test(test_loader, args):  # test the accuracy and fairness of the model
    import datetime
    date_p = datetime.datetime.now()
    date_p.strftime("%Y%m%d%H%M%S")
    timestr = str(date_p).replace(' ', '')
    timestr = timestr.replace('-', '')
    timestr = timestr.replace(':', '')
    timestr = timestr.replace('.', '')
    filepath = str(args.log) + timestr + '.txt'
    logfile = open(filepath, 'w', encoding='utf-8')
    model = modelFactory.find_model(args.model)
    print(args, file=logfile)
    checkpoint = torch.load(args.checkpoint_path)

    model.load_state_dict(checkpointProcess.ProcessCheckpoint(checkpoint), False)
    if args.quantized:
        quantizedModel = quntization(model.to('cpu'))
        return  0
    model.eval()
    correct = 0
    starttime = datetime.datetime.now()
    batchsize = args.batch_size
    correct_list = {}  # correct num in each category
    total_list = {}  # total num of each category
    predicted_list = {}  # predicted num of ecah category
    ff=True
    if args.dataset == 'cifar10':  # calculate the base fairness on cifar 10
        for i in range(10):
            correct_list[i] = 0
            total_list[i] = 0
            predicted_list[i] = 0
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            output = model(data)
            predict = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += predict.eq(target.data.view_as(predict)).cpu().sum()
    endtime = datetime.datetime.now()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('time cost is', (endtime - starttime), 'seconds.', file=logfile)
    print('file size is', os.path.getsize(args.checkpoint_path), 'bytes.', file=logfile)  # print the size of the file
