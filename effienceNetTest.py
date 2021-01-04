from distiller.apputils.data_loaders import load_data
from efficientnet_pytorch import EfficientNet
import torch


def load_imagenet_test_data():
    train_loader, val_loader, test_loader, _ = load_data(dataset='imagenet',
                                                         data_dir='/home/exp/Downloads/xiamutian/imagenet',
                                                         batch_size=8, workers=1)

    loaders = (train_loader, val_loader, test_loader)
    flags = (False, False, True)
    loaders = [loaders[i] for i, flag in enumerate(flags) if flag]

    if len(loaders) == 1:
        # Unpack the list for convinience
        loaders = loaders[0]
    return loaders


if __name__ == '__main__':
    model = EfficientNet.from_pretrained('efficientnet-b1')
    model = model.cuda()
    model.eval()
    test_loader = load_imagenet_test_data()
    correct = 0
    total = 0
    for input, label in test_loader:
        input = input.cuda()
        label = label.cuda()
        outputs = model(input)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == label).sum()
        total = total + len(label)
        print(total, '/', 10000)
    print('accuracy {}'.format(int(correct) / 10000))
