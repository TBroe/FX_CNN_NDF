# Script: FX_CNN_NDF_model
# Author: Tobias Broeckl
# Date: 07.02.2019

# Script defining the python classes for the Convolutional Neural Network (CNN_only) and the Neural Decision Forest.
# An option to support hourly forecast is provided. Default remains the daily_forecast option.

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class FX_FeatureLayer(nn.Sequential):
    def __init__(self, dropout_rate, CNN_only, n_class, daily_forecast):
        super(FX_FeatureLayer, self).__init__()

        self.add_module('conv1', nn.Conv2d(1, 32, kernel_size=5))
        self.add_module('relu1', nn.ReLU())
        self.add_module('pool1', nn.MaxPool2d(kernel_size=2))
        self.add_module('drop1', nn.Dropout(dropout_rate))
        self.add_module('conv2', nn.Conv2d(32, 64, kernel_size=5))
        self.add_module('relu2', nn.ReLU())
        self.add_module('pool2', nn.MaxPool2d(kernel_size=2))
        self.add_module('drop2', nn.Dropout(dropout_rate))

        if CNN_only:
            self.add_module('flatten', Flatten())

            if daily_forecast:
                self.add_module('fc1', nn.Linear(in_features=3 * 3 * 64, out_features=1024))

            else:
                self.add_module('fc1', nn.Linear(in_features=12 * 12 * 64, out_features=1024))

            self.add_module('relu3', nn.ReLU())
            self.add_module('fc2', nn.Linear(in_features=1024, out_features=n_class))

class Flatten(nn.Module):
    def forward(self, x):
        output = x.view(x.size()[0], -1)
        return output

class Tree(nn.Module):
    def __init__(self, depth, n_in_feature, used_feature_rate, n_class):
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.n_class = n_class

        # used features in this tree
        n_used_feature = int(n_in_feature * used_feature_rate)
        onehot = np.eye(n_in_feature)  # [n_in_feature, n_in_feature]
        using_idx = np.random.choice(np.arange(n_in_feature), n_used_feature, replace=False)  # [1, n_in_feature]
        self.feature_mask = onehot[using_idx].T  # [n_in_feature, n_used_feature]
        self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor),
                                       requires_grad=False)
        # leaf label distribution
        self.pi = np.ones((self.n_leaf, n_class)) / n_class
        self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor), requires_grad=False)

        # decision
        self.decision = nn.Sequential(OrderedDict([
           ('linear1', nn.Linear(n_used_feature, self.n_leaf)),
           ('sigmoid', nn.Sigmoid()),
        ]))

    def forward(self, x):
        """
        :param x(Variable): [batch_size,n_features]
        :return: route probability(Variable): [batch_size,n_leaf]
        """
        if x.is_cuda and not self.feature_mask.is_cuda:
            self.feature_mask = self.feature_mask.cuda()

        feats = torch.mm(x, self.feature_mask)  # [batch_size,n_used_feature]
        decision = self.decision(feats)  # [batch_size,n_leaf]

        decision = torch.unsqueeze(decision, dim=2)
        decision_comp = 1 - decision
        decision = torch.cat((decision, decision_comp), dim=2)  # -> [batch_size,n_leaf,2]

        # compute route probability
        # note: we do not use decision[:,0] reason: We only need "leafs-1" decision nodes.
        batch_size = x.size()[0]
        _mu = Variable(
            x.data.new(batch_size, 1, 1).fill_(1.))  # constructs a new tensor of the same data type as self tensor.
        begin_idx = 1
        end_idx = 2
        for n_layer in range(self.depth):
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1,
                                                       2)  # creates prior stage _mu with dimensions for multiplication with the next stage
            _decision = decision[:, begin_idx:end_idx, :]  # [batch_size,2**n_layer,2] #selects the nodes in each layer
            _mu = _mu * _decision  # [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 **(n_layer + 1)

        mu = _mu.view(batch_size, self.n_leaf)

        return mu

    def get_pi(self):

        return self.pi

    def cal_prob(self, mu, pi):
        """
        :param mu [batch_size,n_leaf]
        :param pi [n_leaf,n_class]
        :return: label probability [batch_size,n_class]
        """
        p = torch.mm(mu, pi)
        return p

    def update_pi(self, new_pi):
        self.pi.data = new_pi

class Forest(nn.Module):
    def __init__(self, n_tree, tree_depth, n_in_feature, tree_feature_rate, n_class):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree = n_tree
        for _ in range(n_tree):
            tree = Tree(tree_depth, n_in_feature, tree_feature_rate, n_class)
            self.trees.append(tree)

    def forward(self, x):
        probs = []
        for tree in self.trees:
            mu = tree(x)
            p = tree.cal_prob(mu, tree.get_pi())
            probs.append(p.unsqueeze(2))
        probs = torch.cat(probs, dim=2)
        prob = torch.sum(probs, dim=2) / self.n_tree

        return prob

class NeuralDecisionForest(nn.Module):
    def __init__(self, feature_layer, forest):
        super(NeuralDecisionForest, self).__init__()
        self.feature_layer = feature_layer
        self.forest = forest

    def forward(self, x):
        out = self.feature_layer(x)
        out = out.view(x.size()[0], -1)
        out = self.forest(out)
        return out
