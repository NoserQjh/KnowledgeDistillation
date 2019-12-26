# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:45:48 2019

@author: chxy
"""

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from torchvision.transforms import Resize

import os
import time
import shutil

from tqdm import tqdm
from utils import accuracy, AverageMeter
from resnet import resnet32, resnet110, resnet254
from alexNet import AlexNet
from denseNet import DenseNet121
from googleNet import GoogLeNet
from tensorboard_logger import configure, log_value

from loss import HardDarkRank, RkdDistance, RKdAngle, L2Triplet, AttentionTransfer
import metric.pairsampler as pair
from metric.loss import FitNet
dist_criterion = RkdDistance()
angle_criterion = RKdAngle()
dark_criterion = HardDarkRank(alpha=2, beta=3)
triplet_criterion = L2Triplet(sampler=pair.DistanceWeighted(), margin=0.2)
at_criterion = AttentionTransfer()


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the MobileNet Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        self.config = config

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = config.num_classes

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.weight_decay = config.weight_decay
        self.nesterov = config.nesterov
        self.gamma = config.gamma

        self.triplet_ratio = config.triplet_ratio
        self.dist_ratio = config.dist_ratio
        self.angle_ratio = config.angle_ratio
        self.dark_ratio = config.dark_ratio
        # misc params
        self.use_gpu = config.use_gpu
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.model_name = config.save_name

        self.model_num = config.model_num
        self.models = []
        self.optimizers = []
        self.schedulers = []

        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        self.loss_ce = nn.CrossEntropyLoss()
        self.best_valid_accs = [0.] * self.model_num

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        for i in range(self.model_num):
            # build models
            # def student model
            model = config.model()
            # model = GoogleNet()
            if self.use_gpu:
                model.cuda()

            self.models.append(model)

            # initialize optimizer and scheduler
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov)

            self.optimizers.append(optimizer)

            # set learning rate decay
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizers[i],
                step_size=60,
                gamma=self.gamma,
                last_epoch=-1)
            self.schedulers.append(scheduler)

        print('[*] Number of parameters of one model: {:,}'.format(
            sum([p.data.nelement() for p in self.models[0].parameters()])))

    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)
        else:
            if self.config.use_teacher:
                self.students_init()
        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid))

        for epoch in range(self.start_epoch, self.epochs):

            for scheduler in self.schedulers:
                scheduler.step(epoch)

            print('\nEpoch: {}/{} - LR: {:.6f}'.format(
                epoch + 1,
                self.epochs,
                self.optimizers[0].param_groups[0]['lr'],
            ))

            # train for 1 epoch
            train_losses, train_accs = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_losses, valid_accs = self.validate(epoch)

            for i in range(self.model_num):
                is_best = valid_accs[i].avg > self.best_valid_accs[i]
                msg1 = "model_{:d}: train loss: {:.3f} - train acc: {:.3f} "
                msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
                if is_best:
                    #self.counter = 0
                    msg2 += " [*]"
                msg = msg1 + msg2
                print(
                    msg.format(i + 1, train_losses[i].avg, train_accs[i].avg,
                               valid_losses[i].avg, valid_accs[i].avg))

                # check for improvement
                #if not is_best:
                #self.counter += 1
                #if self.counter > self.train_patience:
                #print("[!] No improvement in a while, stopping training.")
                #return
                self.best_valid_accs[i] = max(valid_accs[i].avg,
                                              self.best_valid_accs[i])
                self.save_checkpoint(
                    i, {
                        'epoch': epoch + 1,
                        'model_state': self.models[i].state_dict(),
                        'optim_state': self.optimizers[i].state_dict(),
                        'best_valid_acc': self.best_valid_accs[i],
                    }, is_best)

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = []
        accs = []

        for i in range(self.model_num):
            self.models[i].train()
            losses.append(AverageMeter())
            accs.append(AverageMeter())

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (images, labels) in enumerate(self.train_loader):
                if self.use_gpu:
                    images, labels = images.cuda(), labels.cuda()
                images, labels = Variable(images), Variable(labels)

                #forward pass
                outputs = []
                for model in self.models:
                    outputs.append(model(images))
                rightList = []
                for i in range(self.model_num):
                    right = torch.max(outputs[i],dim=1)[1]==labels
                    rightList.append(right.int())
                for i in range(self.model_num):
                    ce_loss = self.loss_ce(outputs[i], labels)
                    kl_loss = 0
                    for j in range(self.model_num):
                        if i != j:
                            if self.config.old_loss:
                                kl_loss += 0
                            else:
                                a = torch.mul(torch.transpose(F.log_softmax(outputs[i], dim=1), -1, 0),rightList[j])
                                b = torch.mul(torch.transpose(F.softmax(Variable(outputs[j]), dim=1), -1, 0),rightList[j])
                                kl_loss += self.loss_kl(a,b)
                    if self.model_num <= 1:
                        loss = ce_loss
                    else:
                        loss = ce_loss + kl_loss / (self.model_num - 1)
                    # measure accuracy and record loss
                    prec = accuracy(
                        outputs[i].data, labels.data, topk=(1, ))[0]
                    losses[i].update(loss.item(), images.size()[0])
                    accs[i].update(prec.item(), images.size()[0])

                    # compute gradients and update SGD
                    self.optimizers[i].zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizers[i].step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    ("{:.1f}s - model1_loss: {:.3f} - model1_acc: {:.3f}".
                     format((toc - tic), losses[0].avg, accs[0].avg)))
                self.batch_size = images.shape[0]
                pbar.update(self.batch_size)

                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + i
                    for i in range(self.model_num):
                        log_value('train_loss_%d' % (i + 1), losses[i].avg,
                                  iteration)
                        log_value('train_acc_%d' % (i + 1), accs[i].avg,
                                  iteration)

            return losses, accs

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        losses = []
        accs = []
        for i in range(self.model_num):
            self.models[i].eval()
            losses.append(AverageMeter())
            accs.append(AverageMeter())

        for i, (images, labels) in enumerate(self.valid_loader):
            if self.use_gpu:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            #forward pass
            outputs = []
            for model in self.models:
                outputs.append(model(images))
            ce_losses = []
            for i in range(self.model_num):
                ce_loss = self.loss_ce(outputs[i], labels)
                ce_losses.append(ce_loss)
            ce_mean = torch.mean(torch.stack(ce_losses), 0)
            for i in range(self.model_num):
                ce_loss = self.loss_ce(outputs[i], labels)
                kl_loss = 0
                for j in range(self.model_num):
                    if i != j:
                        '''kl_loss += self.loss_kl(
                            F.log_softmax(outputs[i], dim=1),
                            F.softmax(Variable(outputs[j]),
                                      dim=1)) / ce_losses[j] * ce_mean'''
                        kl_loss += self.loss_kl(
                            F.log_softmax(outputs[i], dim=1),
                            F.softmax(Variable(outputs[j]), dim=1))
                if self.model_num <= 1:
                    loss = ce_loss
                else:
                    loss = ce_loss + kl_loss / (self.model_num - 1)
                # measure accuracy and record loss
                prec = accuracy(outputs[i].data, labels.data, topk=(1, ))[0]
                losses[i].update(loss.item(), images.size()[0])
                accs[i].update(prec.item(), images.size()[0])

        # log to tensorboard for every epoch
        if self.use_tensorboard:
            for i in range(self.model_num):
                log_value('valid_loss_%d' % (i + 1), losses[i].avg, epoch + 1)
                log_value('valid_acc_%d' % (i + 1), accs[i].avg, epoch + 1)

        return losses, accs

    def test(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # load the best checkpoint
        for i in range(self.model_num):
            self.load_checkpoint(i,best=self.best)
            self.models[i].eval()
        for mn in range(self.model_num):
            for i, (images, labels) in enumerate(self.test_loader):
                if self.use_gpu:
                    images, labels = images.cuda(), labels.cuda()
                images, labels = Variable(images), Variable(labels)

                #forward pass
                outputs = self.models[mn](images)
                loss = self.loss_ce(outputs, labels)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
                losses.update(loss.item(), images.size()[0])
                top1.update(prec1.item(), images.size()[0])
                top5.update(prec5.item(), images.size()[0])

            print('[*] Test loss: {:.3f}, top1_acc: {:.3f}%, top5_acc: {:.3f}%'.
                format(losses.avg, top1.avg, top5.avg))

    def save_checkpoint(self, i, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.model_name + str(i + 1) + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + str(i + 1) + '_model_best.pth.tar'
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    def load_checkpoint(self,i, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + str(i + 1) + '_ckpt.pth.tar'
        if best:
            filename = self.model_name + str(i + 1) + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.models[i].load_state_dict(ckpt['model_state'])
        self.optimizers[i].load_state_dict(ckpt['optim_state'])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )

    def students_init(self):
        self.teachers = []
        self.fitnet_criterion = [
            [FitNet(16, 16), FitNet(16, 16), FitNet(32, 32), FitNet(64, 64), FitNet(100,100)],
            [FitNet(16, 16), FitNet(16, 16), FitNet(32, 32), FitNet(64, 64), FitNet(100,100)]]
        [f.cuda() for f in self.fitnet_criterion[0]]
        [f.cuda() for f in self.fitnet_criterion[1]]
        optimizers = []
        lr_schedulers = []
        for i in range(self.model_num):
            if i == 0:
                self.teachers.append(GoogLeNet())
            if i == 1:
                self.teachers.append(DenseNet121())
            ckpt_path = os.path.join(self.ckpt_dir+'/teachers', self.model_name + str(i + 1) + '_model_best.pth.tar')
            ckpt = torch.load(ckpt_path)
            self.teachers[i].load_state_dict(ckpt['model_state'])
            if self.use_gpu:
                self.teachers[i].cuda()
            self.models[i].train()
            self.teachers[i].eval()
            optimizers.append(optim.SGD(
                    self.models[i].parameters(),
                    lr=self.lr,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                    nesterov=self.nesterov))
            lr_schedulers.append(optim.lr_scheduler.MultiStepLR(optimizers[i], milestones=[30,60,90], gamma=0.1))
            lr_schedulers[i].step()

        tic = time.time()
        for ep in range(100):
            loss_all = []
            with tqdm(total=self.num_train) as pbar:
                for i, (images, labels) in enumerate(self.train_loader):
                    if self.use_gpu:
                        images, labels = images.cuda(), labels.cuda()
                    images, labels = Variable(images), Variable(labels)

                    #forward pass
                    outputs = []
                    t_outputs = []
                    for model in self.models:
                        outputs.append(model(images))
                    for model in self.teachers:
                        with torch.no_grad():
                            t_outputs.append(model(images))
                    rightList = []
                    for i in range(self.model_num):
                        right = torch.max(t_outputs[i],dim=1)[1]==labels
                        rightList.append(right.float())
                    for i in range(self.model_num):
                        ce_loss = self.loss_ce(outputs[i], labels)
                        if self.config.old_loss:
                            kl_loss = 0
                        else:
                            a = torch.mul(torch.transpose(F.log_softmax(outputs[i], dim=1), -1, 0),rightList[i])
                            b = torch.mul(torch.transpose(F.softmax(Variable(t_outputs[i]), dim=1), -1, 0),rightList[i])
                            kl_loss = self.loss_kl(a,b)
                        loss = ce_loss + kl_loss

                        optimizers[i].zero_grad()
                        loss.backward()
                        optimizers[i].step()
                        loss_all.append(loss.item())

                        prec = accuracy(outputs[i].data, labels.data, topk=(1, ))[0]

                    pbar.set_description("[Train][Epoch %d] FitNet: %.5f Acc: %.5f " % (ep, loss.item(),prec.item()))
                    self.batch_size = images.shape[0]
                    pbar.update(self.batch_size)
            print('[Epoch %d] Loss: %.5f \n' % (ep, torch.Tensor(loss_all).mean()))
            self.save_checkpoint(
                    i, {
                        'epoch': ep + 1,
                        'model_state': self.models[i].state_dict(),
                        'optim_state': optimizers[i].state_dict(),
                        'best_valid_acc': 0,
                    }, False)

