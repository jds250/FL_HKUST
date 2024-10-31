import torch

from collections import OrderedDict
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import flwr
from flwr_datasets import FederatedDataset

import FedoSSL.open_world_cifar as datasets
from FedoSSL.utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice, cluster_acc_w
import FedoSSL.client_open_world_cifar as client_datasets

NUM_CLIENTS = 2
BATCH_SIZE = 32


def load_datasets():
    #train_label_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), exist_label_list=[0,1,2,3,4,5,6,7,8,9], clients_num=args.clients_num)
    num_classes = 10
    clients_num = 2
    labeled_ratio = 0.5
    labeled_num = 5
    batch_size = 1024
    
    train_label_set = datasets.OPENWORLDCIFAR10(root='FedoSSL/datasets', labeled=True, labeled_num=10, labeled_ratio=labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']))
    # train_unlabel_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), unlabeled_idxs=train_label_set.unlabeled_idxs, exist_label_list=[0,1,2,3,4,5], clients_num=args.clients_num)
    # test_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs, exist_label_list=[0,1,2,3,4,5], clients_num=args.clients_num)
    
    ### prepare clients dataset ###
    ## 子集
    # exist_label_list=[[0,1,2,3,4,5], [0,1,2,3,4,6], [0,1,2,3,4,7], [0,1,2,3,4,8], [0,1,2,3,4,9]]
    # clients_labeled_num = [4, 4, 4, 4, 4]
    ## 全集
    exist_label_list=[[0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,8], [0,1,2,3,4,5,6,8], [0,1,2,3,4,5,6,9]]
    clients_labeled_num = [6, 6, 6, 6, 6]
    ##
    clients_train_label_set = []
    clients_train_unlabel_set = []
    clients_test_set = []
    for i in range(clients_num):
        client_train_label_set = client_datasets.OPENWORLDCIFAR10(root='FedoSSL/datasets', labeled=True, labeled_num=clients_labeled_num[i],
                                                    labeled_ratio=labeled_ratio, download=True,
                                                    transform=TransformTwice(
                                                        datasets.dict_transform['cifar_train']), exist_label_list=exist_label_list[i], clients_num=clients_num)
        client_train_unlabel_set = client_datasets.OPENWORLDCIFAR10(root='FedoSSL/datasets', labeled=False,
                                                        labeled_num=clients_labeled_num[i],
                                                        labeled_ratio=labeled_ratio, download=True,
                                                        transform=TransformTwice(
                                                            datasets.dict_transform['cifar_train']),
                                                        unlabeled_idxs=client_train_label_set.unlabeled_idxs, exist_label_list=exist_label_list[i], clients_num=clients_num)
        # client_test_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num,
        #                                             labeled_ratio=args.labeled_ratio, download=True,
        #                                             transform=datasets.dict_transform['cifar_test'],
        #                                             unlabeled_idxs=client_train_label_set.unlabeled_idxs,
        #                                             exist_label_list=exist_label_list[i],
        #                                             clients_num=args.clients_num)
        client_test_set = client_datasets.OPENWORLDCIFAR10(root='FedoSSL/datasets', labeled=False, labeled_num=labeled_num,
                                                    labeled_ratio=labeled_ratio, download=True,
                                                    transform=datasets.dict_transform['cifar_test'],
                                                    unlabeled_idxs=train_label_set.unlabeled_idxs,
                                                    exist_label_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                    clients_num=clients_num)

        clients_train_label_set.append(client_train_label_set)
        clients_train_unlabel_set.append(client_train_unlabel_set)
        clients_test_set.append(client_test_set)
    ###

    clients_labeled_batch_size = []
    for i in range(clients_num):
        labeled_len = len(clients_train_label_set[i])
        unlabeled_len = len(clients_train_unlabel_set[i])
        labeled_batch_size = int(batch_size * labeled_len / (labeled_len + unlabeled_len))
        clients_labeled_batch_size.append(labeled_batch_size)

    # Initialize the splits  # train_label_loader->client_train_label_loader[];   train_unlabel_loader->client_train_unlabel_loader[]
    client_train_label_loader = []
    client_train_unlabel_loader = []
    client_test_loader = []
    for i in range(clients_num): # train_label_loader->client_train_label_loader[];   train_unlabel_loader->client_train_unlabel_loader[]
        train_label_loader = torch.utils.data.DataLoader(clients_train_label_set[i], batch_size=clients_labeled_batch_size[i], shuffle=True, num_workers=0, drop_last=True)
        train_unlabel_loader = torch.utils.data.DataLoader(clients_train_unlabel_set[i], batch_size=batch_size - clients_labeled_batch_size[i], shuffle=True, num_workers=0, drop_last=True)
        client_train_label_loader.append(train_label_loader)
        client_train_unlabel_loader.append(train_unlabel_loader)

        test_loader = torch.utils.data.DataLoader(clients_test_set[i], batch_size=100, shuffle=False, num_workers=0)
        client_test_loader.append(test_loader)

    return  client_train_label_loader,client_train_unlabel_loader,client_test_loader

