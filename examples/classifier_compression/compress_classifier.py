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
import distiller
from distiller.models import create_model
import distiller.apputils.image_classifier as classifier
import distiller.apputils as apputils
import cmdparser
import os
import numpy as np
import torch

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
        sensitivities = np.arange(*args.sensitivity_range)
        sensitivity_analysis(model, criterion, test_loader, pylogger, args, sensitivities)
        do_exit = True
    elif args.evaluate:
        def print_size_of_model(model):
            import torch
            torch.save(model.state_dict(), "temp.p")
            size = 'Size (MB):' + str(os.path.getsize("temp.p") / 1e6)
            os.remove('temp.p')
            return size

        test_loader = load_test_data(args)

        import torch.quantization as tq
        if args.quantized:
            model = model.to('cpu')
            qmodel = tq.quantize_dynamic(model, inplace=False)
            msglogger.info('model before quantized')
            msglogger.info(print_size_of_model(model))
            msglogger.info('model is quantized')
            msglogger.info(print_size_of_model(qmodel))
            if args.adv != '1':
                classifier.evaluate_model(test_loader, qmodel, criterion, pylogger,
                                      classifier.create_activation_stats_collectors(model, *args.activation_stats),
                                      args, scheduler=compression_scheduler)
                msglogger.info(args.resumed_checkpoint_path)
            import copy
            ADVmodel = copy.deepcopy(model)
            ADVqmodel = qmodel
        else:
            import copy
            if args.adv != '1':
                classifier.evaluate_model(test_loader, model, criterion, pylogger,
                                      classifier.create_activation_stats_collectors(model, *args.activation_stats),
                                      args, scheduler=compression_scheduler)
                msglogger.info(args.resumed_checkpoint_path)
                msglogger.info(print_size_of_model(model))
            ADVmodel = copy.deepcopy(model)

        msglogger.info(args.resumed_checkpoint_path)
        if args.adv == '1':
            ADVmodel.eval()
            testmodel = ADVqmodel if args.quantized else ADVmodel
            # import torchattacks
            # tattack = torchattacks.CW(model=testmodel, c=1)
            # total = 0
            # correct = 0
            # success = 0
            # print('adversrial test')
            # import datetime
            # for data in test_loader:
            #     stime = datetime.datetime.now()
            #     images, labels = data
            #     iamges = images.cuda()
            #     labels = labels.cuda()
            #     adversarial_images = tattack(images, labels)
            #     outputs = testmodel(adversarial_images)
            #     _, adv_predicted = torch.max(outputs.data, 1)
            #     normal_outputs = testmodel(images)
            #     _, normal_predicted = torch.max(normal_outputs.data, 1)
            #     total += labels.size(0)
            #     correct += (adv_predicted == labels).sum()
            #     success += (adv_predicted != normal_predicted).sum()
            #     etime = datetime.datetime.now()
            #     msglogger.info(etime - stime)
            #     msglogger.info('total {} /10000'.format(int(total)))
            #     msglogger.info('correct {}'.format(int(correct)))
            #     msglogger.info('success {}'.format(int(success)))
            #
            # msglogger.info(args.resumed_checkpoint_path)
            # msglogger.info('adversrial success {}%'.format(100*int(success)/int(total)))
            # msglogger.info('adversrial accuracy {}%'.format(100*int(correct)/int(total)))
            import torch.nn as nn
            import torch.optim as optim
            from art.attacks.evasion import CarliniLInfMethod
            from art.classifiers import PyTorchClassifier
            from art.utils import load_cifar10
            from art.utils import load_mnist
            ADVcriterion = nn.CrossEntropyLoss()
            ADVoptimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
            advinput = (3, 32, 32)
            classnum = 10
            print(args.data)
            if 'cifar' in args.data:
                (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10(args.data)
                x_test = (x_test - 0.5) / 0.5
            elif 'mnist' in args.data:
                (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist(args.data)
                x_test = (x_test - 0.1307) / 0.3081
                advinput = (1, 28, 28)
            elif 'imagenet' in args.data:
                import copy
                classnum = 1000
                for validation_step, (inputs, target) in enumerate(test_loader):
                    if validation_step == 0:
                        x_test = copy.deepcopy(inputs).numpy()
                        y_test = copy.deepcopy(target).numpy()
                    else:
                        x_temp = copy.deepcopy(inputs).numpy()
                        y_temp = copy.deepcopy(target).numpy()
                        x_test = np.append(x_test, x_temp, axis=0)
                        y_test = np.append(y_test, y_temp, axis=0)
                        min_pixel_value = 0
                        max_pixel_value = 1
            if 'imagenet' not in args.data:
                x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
                x_test = np.swapaxes(x_test, 2, 3).astype(np.float32)
            ADVclassifier = PyTorchClassifier(model=ADVmodel, clip_values=(min_pixel_value, max_pixel_value),
                                              loss=ADVcriterion,
                                              optimizer=ADVoptimizer, input_shape=advinput, nb_classes=classnum)
            if args.quantized:
                ADVqmodel = ADVqmodel.to('cpu')
                ADVQclassifier = PyTorchClassifier(model=ADVqmodel, clip_values=(min_pixel_value, max_pixel_value),
                                                   loss=ADVcriterion,
                                                   optimizer=ADVoptimizer, input_shape=advinput, nb_classes=classnum)

            # predictions = ADVclassifier.predict(x_test, batch_size=args.batch_size)
            import torchnet.meter as tnt
            # if 'imagenet' in args.data:
            #     classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
            #     classerr.add(predictions, y_test)
            #     accuracy = classerr.value()[0]
            # else:
            #     accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            import datetime
            attack = CarliniLInfMethod(classifier=ADVclassifier, batch_size=64, learning_rate=0.05)
            x_test_adv = attack.generate(x=x_test[:100])
            if args.quantized:
                x_test_adv_tensor = torch.from_numpy(x_test_adv)
                y_test_temp = np.where(y_test==1)[1]
                y_test_adv_tensor = torch.from_numpy(y_test_temp).to('cpu')
                x_test_adv_tensor = x_test_adv_tensor.to('cpu')
                ADVqmodel = ADVqmodel.to('cpu')
                batch_size = args.batch_size
                bathc_range = x_test_adv_tensor.shape[0]/batch_size
                total = 0
                correct = 0
                for batch in range(int(bathc_range+1)):
                    if (batch+1)*batch_size<x_test_adv_tensor.shape[0]:
                        test_x = x_test_adv_tensor[batch*batch_size:(batch+1)*batch_size,:,:,:]
                        test_y = y_test_adv_tensor[batch*batch_size:(batch+1)*batch_size]
                    else:
                        test_x = x_test_adv_tensor[batch*batch_size:,:,:,:]
                        test_y = y_test_adv_tensor[batch*batch_size:]
                    outputs = ADVqmodel(test_x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += test_x.size(0)
                    correct += (predicted == test_y).sum()
                    print('correct',int(correct))
                    print('total',int(total))
                msglogger.info('adversiral accuracy{}%'.format(int(correct)/int(total)))
                # predictions = ADVQclassifier.predict(x_test_adv, batch_size=args.batch_size)
            else:
                x_test_adv_tensor = torch.from_numpy(x_test_adv)
                y_test_temp = np.where(y_test==1)[1]
                y_test_adv_tensor = torch.from_numpy(y_test_temp)
                ADVmodel = ADVmodel.cuda()
                batch_size = args.batch_size
                bathc_range = x_test_adv_tensor.shape[0]/batch_size
                total = 0
                correct = 0
                for batch in range(int(bathc_range+1)):
                    if (batch+1)*batch_size<x_test_adv_tensor.shape[0]:
                        test_x = x_test_adv_tensor[batch*batch_size:(batch+1)*batch_size,:,:,:]
                        test_y = y_test_adv_tensor[batch*batch_size:(batch+1)*batch_size]
                    else:
                        test_x = x_test_adv_tensor[batch*batch_size:,:,:,:]
                        test_y = y_test_adv_tensor[batch*batch_size:]
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()
                    outputs = ADVmodel(test_x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += test_x.size(0)
                    correct += (predicted == test_y).sum()

                msglogger.info('Accuracy on adversarial test examples self test:{}%'.format(100*int(correct)/int(total)))
                print('s')
                predictions = ADVclassifier.predict(x_test_adv, batch_size=args.batch_size)
            if 'imagenet' in args.data:
                classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
                classerr.add(predictions, y_test)
                accuracy = classerr.value()[0]
            else:
                accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            msglogger.info(args.resumed_checkpoint_path)
            msglogger.info('Accuracy on adversarial test examples: {}%'.format(100*accuracy))
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
