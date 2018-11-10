from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
    
    
class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.BCELoss()):#CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
        
    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        # train part of the model
        #optim = self.optim(model.classifier.parameters(), **self.optim_args)
        # or everything
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model=model.cuda()
        print('START TRAIN.')
        # define a loss function for classification
        lossFunction = torch.nn.CrossEntropyLoss()#torch.nn.BCELoss()

        # main training loop
        iter = 0
        for epoch in range(num_epochs):
            # reset the variables, set model into train mode (important for dropout)
            correct_training_overall = 0
            total_training = 0
            self.train_loss_history.append(0)
            model.train(True)
            # run a training epoch
            for batch_num, (data, target) in enumerate(train_loader):
                # update step for a mini-batch
                if torch.cuda.is_available():
                    data, target = Variable(data.cuda()) ,Variable(target.cuda())
                else:
                    data, target = Variable(data), Variable(target)
                optim.zero_grad()
                output = model(data)
                loss = lossFunction(output, target)
                loss.backward()
                optim.step()
                # logging
                iter += train_loader.batch_size
                if iter % (10**log_nth) == 0:
                    print('[Iteration %d/%d] TRAIN loss: %.3f' % 
                            (iter, iter_per_epoch*num_epochs*train_loader.batch_size, loss.data[0]))
                # collect statistics
                print(output.data)
                _, predicted = torch.max(output.data,1)
                correct_training = (predicted == target.data).sum()
                correct_training_overall += correct_training
                total_training += len(predicted) 
                self.train_loss_history[-1] += loss.data[0] 
             
            # print the statistics   
            self.train_acc_history.append( float(correct_training_overall) / total_training)
            self.train_loss_history[-1] /= len(train_loader)    
            print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % 
                            (epoch, num_epochs, self.train_acc_history[-1], self.train_loss_history[-1]))
             
            print('Correctly classified SAMPLES in this epoch: %d' % correct_training_overall)
            # run accuracy and loss checks on the validation set
            correct_validation = 0
            total_validation = 0
            self.val_loss_history.append(0)
            # set into test-mode (e.g. dropout)
            model.train(False)
            for batch_num, (data, target) in enumerate(val_loader):
                 if torch.cuda.is_available():
                     data, target = Variable(data.cuda()), Variable(target.cuda())
                 else:
                     data, target = Variable(data), Variable(target)
                 output = model(data)
                 loss = lossFunction(output, target)
                 _, predicted = torch.max(output.data,1)
                 correct_validation += (predicted == target.data).sum()
                 total_validation += len(target)
                 self.val_loss_history[-1] += loss.data[0] 
            # print statistics on the validation set
            self.val_loss_history[-1] /= len(val_loader)    
            self.val_acc_history.append(float(correct_validation) / total_validation)
            print('[Epoch %d/%d] VAL acc/loss: %.3f/%.3f' % 
                            (epoch, num_epochs, self.val_acc_history[-1], self.val_loss_history[-1]))
            
        print('FINISH.')
