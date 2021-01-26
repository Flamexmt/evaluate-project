#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""This is an example application for compressing image classification models.

The application borrows its main flow code from torchvision's ImageNet classification
training sample application (https://github.com/pytorch/examples/tree/master/imagenet).
We tried to keep it similar, in order to make it familiar and easy to understand.

Integrating compression is very simple: simply add invocations of the appropriate
compression_scheduler callbacks, for each stage in the training.  The training skeleton
looks like the pseudo code below.  The boiler-plate Pytorch classification training
is speckled with invocations of CompressionScheduler.

For each epoch:
    compression_scheduler.on_epoch_begin(epoch)
    train()
    validate()
    compression_scheduler.on_epoch_end(epoch)
    save_checkpoint()

train():
    For each training step:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        compression_scheduler.before_parameter_optimization(epoch)
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)


This exmple application can be used with torchvision's ImageNet image classification
models, or with the provided sample models:

- ResNet for CIFAR: https://github.com/junyuseu/pytorch-cifar-models
- MobileNet for ImageNet: https://github.com/marvis/pytorch-mobilenet
"""

import traceback
import logging
from functools import partial

from art.utils import load_cifar10, load_mnist

import distiller
from distiller.models import create_model
import distiller.apputils.image_classifier as classifier
import distiller.apputils as apputils
import cmdparser
import os
import numpy as np
import torch

from torch import nn

import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.qat as nniqat
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.qat as nnqat
from torch.quantization.stubs import QuantStub, DeQuantStub

# Map for swapping float module to quantized ones
# Logger handle
msglogger = logging.getLogger()


def main():
    # Parse arguments
    args = cmdparser.add_cmdline_args(classifier.init_classifier_compression_arg_parser()).parse_args()
    app = ClassifierCompressorSampleApp(args, script_dir=os.path.dirname(__file__))
    if app.handle_subapps():
        return
    init_knowledge_distillation(app.args, app.model, app.compression_scheduler)
    app.run_training_loop()  # mark
    # Finally run results on the test set
    return app.test()


def handle_subapps(model, criterion, optimizer, compression_scheduler, pylogger, args):
    def load_test_data(args):
        test_loader = classifier.load_data(args, load_train=False, load_val=False, load_test=True)
        return test_loader

    def load_train_data(args):
        train_loader = classifier.load_data(args, load_train=True, load_val=False, load_test=False)
        return train_loader

    do_exit = False
    if args.greedy:
        greedy(model, criterion, optimizer, pylogger, args)
        do_exit = True


    elif args.summary:
        # This sample application can be invoked to produce various summary reports
        for summary in args.summary:
            distiller.model_summary(model, summary, args.dataset)
        do_exit = True
    elif args.export_onnx is not None:
        distiller.export_img_classifier_to_onnx(model,
                                                os.path.join(msglogger.logdir, args.export_onnx),
                                                args.dataset, add_softmax=True, verbose=False)
        do_exit = True
    elif args.qe_calibration and not (args.evaluate and args.quantize_eval):
        classifier.acts_quant_stats_collection(model, criterion, pylogger, args, save_to_file=True)
        do_exit = True
    elif args.activation_histograms:
        classifier.acts_histogram_collection(model, criterion, pylogger, args)
        do_exit = True
    elif args.sensitivity is not None:
        test_loader = load_test_data(args)
        import numpy as np
        sensitivities = np.arange(*args.sensitivity_range)
        sensitivity_analysis(model, criterion, test_loader, pylogger, args, sensitivities)
        do_exit = True
    elif args.evaluate:
        def print_size_of_model(model):
            import torch
            temp = model.state_dict()
            torch.save(temp, "temp.p")
            size = 'Size (MB):' + str(os.path.getsize("temp.p") / 1e6)
            os.remove('temp.p')
            return size
        test_loader = load_test_data(args)

        import torch.quantization as tq
        from torch import nn
        # Map for swapping float module to quantized ones

        if args.quantized == '8':
            model = model.to(torch.device('cpu'))
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            import copy
            model.eval()
            if 'imagenet' in args.data:
                quantization_config = torch.quantization.get_default_qconfig("fbgemm")
                fused_model = model.fuse_model()
                fused_model.qconfig = quantization_config
            else:
                fused_model = copy.deepcopy(model)
                fused_model = fused_model.quantize_self()
                fused_model.eval()
                fused_model.fuse_self()
            model_fp32_prepared = torch.quantization.prepare(fused_model, inplace=False)
            model_fp32_prepared.eval()

            def calibrate_model(model, loader):
                print('start calibrate!!')
                model.eval()
                i = 0
                for inputs, labels in loader:
                    _ = model(inputs)
                    print(i, '/', len(loader))
                    i += 1
                    break
            calibrate_data = test_loader
            calibrate_model(model_fp32_prepared, calibrate_data)
            model_int8 = torch.quantization.convert(model_fp32_prepared)
            from distiller.apputils.checkpoint import save_checkpoint
            save_checkpoint(epoch=0, model=model_int8, arch=args.arch, name='quantized_' + args.arch,
                            dir='../../outputsdata/eval/quantized_models/', extras={'quantized': True})
            model_int8.eval()
            msglogger.info('model before quantized')
            msglogger.info(print_size_of_model(model))
            msglogger.info('model is quantized')
            msglogger.info(print_size_of_model(model_int8))
            print('float')
            classifier.evaluate_model(test_loader, model, criterion, pylogger,
                                      classifier.create_activation_stats_collectors(model,
                                                                                    *args.activation_stats),
                                      args, scheduler=compression_scheduler)
            print('int')
            classifier.evaluate_model(test_loader, model_int8, criterion, pylogger,
                                      classifier.create_activation_stats_collectors(model_int8,
                                                                                    *args.activation_stats),
                                      args, scheduler=compression_scheduler)
            msglogger.info(args.resumed_checkpoint_path)
        elif args.quantized == '16':
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            # model.fuse_model()
            msglogger.info('model before half')
            msglogger.info(print_size_of_model(model))
            model.half()
            msglogger.info('model is half')
            msglogger.info(print_size_of_model(model))
            from distiller.apputils.checkpoint import save_checkpoint
            save_checkpoint(epoch=0, model=model, arch=args.arch, name='half_' + args.arch,
                            dir='../../outputsdata/eval/quantized_models/', extras={'half_': True})
            calibrate_data = test_loader

            # save point
            model.eval()

            print('float 16')
            correct = 0
            for input, label in test_loader:
                input = input.half()
                input = input.cuda()
                label = label.cuda()
                outputs = model(input)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == label).sum()
            msglogger.info('accuracy {}'.format(int(correct) / int(10000)))
            msglogger.info(args.resumed_checkpoint_path)
        elif args.fairness_test =='1':
            import  pickle
            import torchvision.transforms as transforms
            from  distiller.apputils.data_loaders import GrayCifarDataset
            if 'cifar' in args.arch:
                data_setting = {
                    'test_color_path': '../data.cifar/cifar_color_test_imgs',
                    'test_gray_path': './data.cifar/cifar_gray_test_imgs',
                    'test_label_path': './data.cifar/cifar_test_labels',
                }
            else:
                data_setting = {}
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            test_color_data = GrayCifarDataset(data_setting['test_color_path'],
                                                      data_setting['test_label_path'],
                                                      transform_test)
            test_gray_data = GrayCifarDataset(data_setting['test_gray_path'],
                                                     data_setting['test_label_path'],
                                                     transform_test)


            test_color_loader = torch.utils.data.DataLoader(
                test_color_data, batch_size=args.batch_size,
                shuffle=False, num_workers=2)
            test_gray_loader = torch.utils.data.DataLoader(
                test_gray_data, batch_size=args.batch_size,
                shuffle=False, num_workers=2)


            print('do normal testing')
            color_correct = 0
            color_total = 0
            for i, (input, label) in enumerate(test_color_loader):
                outputs = model(input)
                _, predicted = torch.max(outputs.data, 1)
                color_correct += (predicted == label).sum()
                color_total += len(label)
            normal_accuracy =int(color_correct) / int(color_total)
            print('accuracy at normal images {}'.format(normal_accuracy))

            gray_correct = 0
            gray_total = 0
            for i, (input, label) in enumerate(test_gray_loader):
                outputs = model(input)
                _, predicted = torch.max(outputs.data, 1)
                gray_correct += (predicted == label).sum()
                gray_total += len(label)
            gray_accuracy =  int(gray_correct) / int(gray_total)
            print('accuracy at gray images {}'.format(gray_accuracy))


            pass
        else:
            if args.adv != '1':
                import copy
                # model.fuse_model()
                classifier.evaluate_model(test_loader, model, criterion, pylogger,
                                          classifier.create_activation_stats_collectors(model, *args.activation_stats),
                                          args, scheduler=compression_scheduler)
                msglogger.info(print_size_of_model(model))
                # ADVmodel = copy.deepcopy(model)
        if args.adv == '1':
            import numpy as np
            import torch.nn as nn
            import art.config
            import torch.optim as optim
            from  art.classifiers import PyTorchClassifier

            # prepare tester
            ADVcriterion = nn.CrossEntropyLoss()
            # load data
            advinput = (3, 32, 32)
            classnum = 10
            if 'cifar' in args.data:
                art.config.ART_DATA_PATH = args.data
                (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
                x_test = (x_test - 0.5) / 0.5
            elif 'mnist' in args.data:
                art.config.ART_DATA_PATH = args.data
                (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
                x_test = (x_test - 0.1307) / 0.3081
                advinput = (1, 28, 28)
            elif 'imagenet' in args.data:
                if args.numpy_xpath != '':
                    print('load data from file')
                    x_test = np.load(args.numpy_xpath)
                    y_test = np.load(args.numpy_ypath)
                else:
                    import copy
                    classnum = 1000
                    advinput = (3, 224, 224)
                    min_pixel_value = 0
                    max_pixel_value = 1
                    print('tensor to numpy process')
                    for validation_step, (inputs, target) in enumerate(test_loader):
                        if validation_step == 0:
                            x_test = (inputs).numpy()
                            y_test = (target).numpy()
                            break
                        else:
                            x_temp = (inputs).numpy()
                            y_temp = (target).numpy()
                            x_test = np.append(x_test, x_temp, axis=0)
                            y_test = np.append(y_test, y_temp, axis=0)
                    np.save('imagenet_x_test',x_test)
                    np.save('imagenet_y_test',y_test)
            if 'imagenet' not in args.data:
                x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
                x_test = np.swapaxes(x_test, 2, 3).astype(np.float32)

            # model = model.to('cpu')
            if 'half' in args.resumed_checkpoint_path:
                ADVclassifier = PyTorchClassifier(model=model.half(), clip_values=(min_pixel_value, max_pixel_value),
                                              loss=ADVcriterion, input_shape=advinput, nb_classes=classnum)
                msglogger.info('half model!')
            elif 'quantized' in args.resumed_checkpoint_path:
                ADVclassifier = PyTorchClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value),
                                                  loss=ADVcriterion, input_shape=advinput, nb_classes=classnum)
                msglogger.info('quantized model!')
            else:
                ADVclassifier = PyTorchClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value),
                                                  loss=ADVcriterion, input_shape=advinput, nb_classes=classnum)
                msglogger.info('normal model!')
            x_test = x_test[:]
            y_test = y_test[:]
            msglogger.info('do normal test')
            normal_predictions = ADVclassifier.predict(x_test,batch_size=args.batch_size)
            if 'imagenet' in args.data:
                normal_predictions = torch.nn.Softmax(dim=1)(torch.from_numpy(normal_predictions))
                normal_predictions = np.argmax(normal_predictions, axis=1)
                normal_predictions = normal_predictions.numpy()
                t = (np.where((normal_predictions ) == (y_test)))[0]
                accuracy = len(t)/ len(y_test)
            else:
                accuracy = np.sum(np.argmax(normal_predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            msglogger.info("Accuracy on normal test: {}%".format(accuracy * 100))

            if args.cw_attack == '1':
                msglogger.info('--------------------')
                msglogger.info('do CarliniL2Method Attack test!')
                from art.attacks.evasion import CarliniL2Method
                cw_attack = CarliniL2Method(classifier=ADVclassifier, batch_size=args.adv_batch_size,binary_search_steps=20,max_iter=10)
                x_test_adv = cw_attack.generate(x=x_test[:])
                msglogger.info('success generate CarliniL2Method Attack')
                predictions = ADVclassifier.predict(x_test_adv, batch_size=args.adv_batch_size)
                if 'imagenet' in args.data:
                    predictions = torch.nn.Softmax(dim=1)(torch.from_numpy(predictions))
                    predictions = np.argmax(predictions, axis=1)
                    predictions = predictions.numpy()
                    accuracy = len((np.where((predictions) == (y_test)))[0]) / len(y_test)
                    success_rate = len((np.where((predictions) != (normal_predictions)))[0]) / len(y_test)
                else:
                    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
                    success_rate = np.sum(
                        np.argmax(predictions, axis=1) != np.argmax(normal_predictions, axis=1)) / len(
                        y_test)
                msglogger.info("Accuracy under cw2 Attack: {}%".format(accuracy * 100))

                msglogger.info("Attack sucess of cw2 Attack: {}%".format(success_rate * 100))
                msglogger.info("{}%/{}%".format(accuracy * 100, success_rate * 100))


            if args.pgd_attack == '1':

                msglogger.info('--------------------')
                msglogger.info('do PGD test!')
                from art.attacks.evasion import ProjectedGradientDescent
                pgd_attack = ProjectedGradientDescent(estimator=ADVclassifier, batch_size=args.adv_batch_size,eps=8,eps_step=2)
                x_test_adv = pgd_attack.generate(x=x_test)
                msglogger.info('success generate ProjectedGradientDescent Attack')
                predictions = ADVclassifier.predict(x_test_adv,batch_size=args.adv_batch_size)
                if 'imagenet' in args.data:
                    predictions = torch.nn.Softmax(dim=1)(torch.from_numpy(predictions))
                    predictions = np.argmax(predictions, axis=1)
                    predictions = predictions.numpy()
                    accuracy = len((np.where((predictions) == (y_test)))[0]) / len(y_test)
                    success_rate = len((np.where((predictions) != (normal_predictions)))[0]) / len(y_test)
                else:
                    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
                    success_rate = np.sum(
                        np.argmax(predictions, axis=1) != np.argmax(normal_predictions, axis=1)) / len(
                        y_test)
                msglogger.info("Accuracy under ProjectedGradientDescent Attack: {}%".format(accuracy * 100))

                msglogger.info("Attack sucess of ProjectedGradientDescent Attack: {}%".format(success_rate * 100))
                msglogger.info("{}%/{}%".format(accuracy * 100,success_rate * 100))
                pass
            if args.square_attack == '1':
                msglogger.info('--------------------')
                msglogger.info('do SquareAttack test!')
                from art.attacks.evasion.square_attack import SquareAttack
                square_attack = SquareAttack(estimator=ADVclassifier, batch_size=args.adv_batch_size, p_init=0.3)
                x_test_adv = square_attack.generate(x=x_test)
                msglogger.info('success generate SquareAttack')
                predictions = ADVclassifier.predict(x_test_adv, batch_size=args.batch_size)
                if 'imagenet' in args.data:
                    predictions = torch.nn.Softmax(dim=1)(torch.from_numpy(predictions))
                    predictions = np.argmax(predictions, axis=1)
                    predictions = predictions.numpy()
                    accuracy = len((np.where((predictions) == (y_test)))[0]) / len(y_test)
                    success_rate = len((np.where((predictions) != (normal_predictions)))[0]) / len(y_test)
                else:
                    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
                    success_rate = np.sum(
                        np.argmax(predictions, axis=1) != np.argmax(normal_predictions, axis=1)) / len(
                        y_test)
                msglogger.info("Accuracy under SquareAttack: {}%".format(accuracy * 100))
                msglogger.info("Attack sucess of SquareAttack Attack: {}%".format(success_rate * 100))
                msglogger.info("{}%/{}%".format(accuracy * 100, success_rate * 100))


            if args.extraction_attack == '1':
                msglogger.info('--------------------')
                msglogger.info('do Extraction Attack test!')
                from art.attacks.extraction.knockoff_nets import KnockoffNets
                epochs = args.extraction_epoch
                msglogger.info('epoch is')
                msglogger.info(str(epochs))
                extraction_attack = KnockoffNets(classifier=ADVclassifier, nb_epochs=epochs, nb_stolen=10000,
                                                 batch_size_fit=args.batch_size)
                cifar_model = distiller.models.create_model(args.pretrained, args.dataset, args.arch,
                                                            parallel=not args.load_serialized, device_ids=args.gpus)
                thief_optimizer = torch.optim.SGD(cifar_model.parameters(), lr=0.01)
                if 'half' in args.resumed_checkpoint_path:
                    thief_classifier = art.classifiers.PyTorchClassifier(model=cifar_model.half(),
                                                                         optimizer=thief_optimizer,
                                                                         clip_values=(min_pixel_value, max_pixel_value),
                                                                         loss=ADVcriterion, input_shape=advinput,
                                                                         nb_classes=classnum)
                    msglogger.info('half model!')
                elif 'quantized' in args.resumed_checkpoint_path:
                    thief_classifier = art.classifiers.PyTorchClassifier(model=cifar_model,
                                                                         optimizer=thief_optimizer,
                                                                         clip_values=(min_pixel_value, max_pixel_value),
                                                                         loss=ADVcriterion, input_shape=advinput,
                                                                         nb_classes=classnum)
                    msglogger.info('quantized model!')
                else:
                    thief_classifier = art.classifiers.PyTorchClassifier(model=cifar_model,
                                                                         optimizer=thief_optimizer,
                                                                         clip_values=(min_pixel_value, max_pixel_value),
                                                                         loss=ADVcriterion, input_shape=advinput,
                                                                         nb_classes=classnum)
                    msglogger.info('normal model!')

                black_box_model = extraction_attack.extract(x=x_test[:], thieved_classifier=thief_classifier)
                msglogger.info('success generate Extraction Attack')
                y_test_predicted_extracted = black_box_model.predict(x_test)
                y_test_predicted_target = ADVclassifier.predict(x_test)

                format_string = np.sum(np.argmax(y_test_predicted_target, axis=1) == np.argmax(y_test, axis=1)) / \
                                y_test.shape[0]

                msglogger.info("Victime model - Test accuracy:")
                msglogger.info(str(format_string))
                format_string = np.sum(np.argmax(y_test_predicted_extracted, axis=1) == np.argmax(y_test, axis=1)) / \
                                y_test.shape[0]
                msglogger.info(
                    "Extracted model - Test accuracy:")
                msglogger.info(str(format_string))
                accuracy = np.sum(np.argmax(y_test_predicted_extracted, axis=1) == np.argmax(y_test, axis=1)) / \
                           y_test.shape[0]
                format_string = np.sum(
                    np.argmax(y_test_predicted_extracted, axis=1) == np.argmax(y_test_predicted_target, axis=1)) / \
                                y_test_predicted_target.shape[0]
                msglogger.info(
                    "Extracted model - Test Fidelity:")
                msglogger.info(str(format_string))
                success_rate = np.sum(
                    np.argmax(y_test_predicted_extracted, axis=1) == np.argmax(y_test_predicted_target, axis=1)) / \
                               y_test_predicted_target.shape[0]
            msglogger.info(args.resumed_checkpoint_path)
            msglogger.info("{}%/{}%".format(accuracy * 100, success_rate * 100))

        do_exit = True
    elif args.thinnify:
        assert args.resumed_checkpoint_path is not None, \
            "You must use --resume-from to provide a checkpoint file to thinnify"
        distiller.contract_model(model, compression_scheduler.zeros_mask_dict, args.arch, args.dataset, optimizer=None)
        apputils.save_checkpoint(0, args.arch, model, optimizer=None, scheduler=compression_scheduler,
                                 name="{}_thinned".format(args.resumed_checkpoint_path.replace(".pth.tar", "")),
                                 dir=msglogger.logdir)
        msglogger.info("Note: if your model collapsed to random inference, you may want to fine-tune")
        do_exit = True
    return do_exit


def init_knowledge_distillation(args, model, compression_scheduler):
    args.kd_policy = None
    if args.kd_teacher:
        teacher = create_model(args.kd_pretrained, args.dataset, args.kd_teacher, device_ids=args.gpus)
        if args.kd_resume:
            teacher = apputils.load_lean_checkpoint(teacher, args.kd_resume)
        dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt, args.kd_teacher_wt)
        args.kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher, args.kd_temp, dlw)
        compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch, ending_epoch=args.epochs,
                                         frequency=1)
        msglogger.info('\nStudent-Teacher knowledge distillation enabled:')
        msglogger.info('\tTeacher Model: %s', args.kd_teacher)
        msglogger.info('\tTemperature: %s', args.kd_temp)
        msglogger.info('\tLoss Weights (distillation | student | teacher): %s',
                       ' | '.join(['{:.2f}'.format(val) for val in dlw]))
        msglogger.info('\tStarting from Epoch: %s', args.kd_start_epoch)


def early_exit_init(args):
    if not args.earlyexit_thresholds:
        return
    args.num_exits = len(args.earlyexit_thresholds) + 1
    args.loss_exits = [0] * args.num_exits
    args.losses_exits = []
    args.exiterrors = []
    msglogger.info('=> using early-exit threshold values of %s', args.earlyexit_thresholds)


class ClassifierCompressorSampleApp(classifier.ClassifierCompressor):
    def __init__(self, args, script_dir):
        print(args)
        super().__init__(args, script_dir)
        early_exit_init(self.args)
        # Save the randomly-initialized model before training (useful for lottery-ticket method)
        if args.save_untrained_model:
            ckpt_name = '_'.join((self.args.name or "", "untrained"))
            apputils.save_checkpoint(0, self.args.arch, self.model,
                                     name=ckpt_name, dir=msglogger.logdir)

    def handle_subapps(self):
        return handle_subapps(self.model, self.criterion, self.optimizer,
                              self.compression_scheduler, self.pylogger, self.args)


def sensitivity_analysis(model, criterion, data_loader, loggers, args, sparsities):
    # This sample application can be invoked to execute Sensitivity Analysis on your
    # model.  The ouptut is saved to CSV and PNG.
    msglogger.info("Running sensitivity tests")
    if not isinstance(loggers, list):
        loggers = [loggers]
    test_fnc = partial(classifier.test, test_loader=data_loader, criterion=criterion,
                       loggers=loggers, args=args,
                       activations_collectors=classifier.create_activation_stats_collectors(model))
    which_params = [param_name for param_name, _ in model.named_parameters()]
    sensitivity = distiller.perform_sensitivity_analysis(model,
                                                         net_params=which_params,
                                                         sparsities=sparsities,
                                                         test_func=test_fnc,
                                                         group=args.sensitivity)
    distiller.sensitivities_to_png(sensitivity, os.path.join(msglogger.logdir, 'sensitivity.png'))
    distiller.sensitivities_to_csv(sensitivity, os.path.join(msglogger.logdir, 'sensitivity.csv'))


def greedy(model, criterion, optimizer, loggers, args):
    train_loader, val_loader, test_loader = classifier.load_data(args)

    test_fn = partial(classifier.test, test_loader=test_loader, criterion=criterion,
                      loggers=loggers, args=args, activations_collectors=None)
    train_fn = partial(classifier.train, train_loader=train_loader, criterion=criterion, args=args)
    assert args.greedy_target_density is not None
    distiller.pruning.greedy_filter_pruning.greedy_pruner(model, args,
                                                          args.greedy_target_density,
                                                          args.greedy_pruning_step,
                                                          test_fn, train_fn)


def start():
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None and hasattr(msglogger, 'log_filename'):
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None and hasattr(msglogger, 'log_filename'):
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
