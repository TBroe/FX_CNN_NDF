# script: classes.py
# author: Tobias Broeckl
# date: 07.05.2019

# This script defines all classes that are used to implement both the deep CNN and deep NDF.
# These classes are referenced in both the model selection and model evaluation scripts.
# Script is best understood in context of the thesis appendix "Model Implementations in Python".

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class CNN_only(nn.Module):
    def __init__(self):
        super(CNN_only, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(in_features=64 * 3 * 3, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(x), 2)
        x = self.conv2(x)
        x = F.dropout2d(x, p=0.5,training=self.training)
        x = F.max_pool2d(F.relu(x), 2)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class CNN_forest(nn.Module):
    def __init__(self):
        super(CNN_forest, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(in_features=64 * 3 * 3, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(x), 2)
        x = self.conv2(x)
        x = F.dropout2d(x, p=0.5, training=self.training)
        x = F.max_pool2d(F.relu(x), 2)

        return x

class Tree(nn.Module):
    def __init__(self, depth, n_in_feature, used_feature_rate):
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth

        n_used_feature = int(n_in_feature * used_feature_rate)
        onehot = np.eye(n_in_feature)
        np.random.seed(0)
        using_idx = np.random.choice(np.arange(n_in_feature), n_used_feature, replace=False)
        self.feature_mask = onehot[using_idx].T
        self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor), requires_grad=False)

        self.pi = np.ones((self.n_leaf, 2)) / 2
        self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor), requires_grad=False)

        self.decision = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(in_features=n_used_feature, out_features=self.n_leaf)),
            ('sigmoid', nn.Sigmoid()),
        ]))

    def forward(self, x):
        if x.is_cuda and not self.feature_mask.is_cuda:
            self.feature_mask = self.feature_mask.cuda()

        feats = torch.mm(x, self.feature_mask)
        decision = self.decision(feats)

        decision = torch.unsqueeze(decision, dim=2)
        decision_comp = 1 - decision
        decision = torch.cat((decision, decision_comp), dim=2)

        batch_size = x.size()[0]
        _mu = Variable(x.data.new(batch_size, 1, 1).fill_(1.))
        begin_idx = 1
        end_idx = 2
        for n_layer in range(self.depth):
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _decision = decision[:, begin_idx:end_idx, :]
            _mu = _mu * _decision
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer + 1)

        mu = _mu.view(batch_size, self.n_leaf)

        return mu

    def get_pi(self):

        return self.pi

    def cal_prob(self, mu, pi):
        p = torch.mm(mu, pi)
        return p

    def update_pi(self, new_pi):
        self.pi.data = new_pi


class Forest(nn.Module):
    def __init__(self, n_tree, tree_depth, n_in_feature, tree_feature_rate):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree = n_tree
        for _ in range(n_tree):
            tree = Tree(tree_depth, n_in_feature, tree_feature_rate)
            self.trees.append(tree)

    def forward(self, x):
        probs = []
        for tree in self.trees:
            mu = tree(x)
            p = torch.mm(mu, tree.pi)
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
