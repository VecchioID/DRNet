# -*- coding:utf-8 -*-
# __author__ = 'Vecchio'
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self, args):
        super(BasicModel, self).__init__()
        self.name = args.save_model_name

    def load_model(self, path, epoch):
        state_dict = torch.load(path + '{}_epoch_{}.pth'.format(self.name, epoch))['state_dict']
        self.load_state_dict(state_dict)

    def save_model(self, path, epoch, acc, loss):
        torch.save({'state_dict': self.state_dict(), 'acc': acc, 'loss': loss},
                   path + '{}_epoch_{}_acc_{}.pth'.format(self.name, epoch, acc))

    def compute_loss(self, output, target):
        pass

    def train_(self, image, target):
        self.optimizer.zero_grad()
        output = self(image)
        loss = self.compute_loss(output, target)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def validate_(self, image, target):
        with torch.no_grad():
            output = self(image)
        loss = self.compute_loss(output, target)
        pred = output.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def test_(self, image, target):
        with torch.no_grad():
            output = self(image)
        pred = output.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return accuracy
