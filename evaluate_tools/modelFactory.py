import distiller.models.cifar10 as cifar10_model
import distiller.models.mnist as mnist_model
import distiller.models.imagenet as imagenet_model
def find_model(modelname):
    if modelname=='simple_mnist':
        return mnist_model.simplenet_mnist()
    elif modelname=='plain20_cifar':
        return cifar10_model.plain20_cifar()
    elif modelname=='resnet20_cifar':
        return cifar10_model.resnet20_cifar()
    elif modelname=='resnet32_cifar':
        return cifar10_model.resnet32_cifar()
    elif modelname=='resnet44_cifar':
        return cifar10_model.resnet44_cifar()
    elif modelname=='resnet56_cifar':
        return cifar10_model.resnet56_cifar()
    elif modelname=='plain20_cifar':
        return cifar10_model.plain20_cifar()
    elif modelname=='plain20_cifar':
        return cifar10_model.plain20_cifar()
    elif modelname=='plain20_cifar':
        return cifar10_model.plain20_cifar()
    elif modelname=='plain20_cifar':
        return cifar10_model.plain20_cifar()
    elif modelname=='plain20_cifar':
        return cifar10_model.plain20_cifar()
