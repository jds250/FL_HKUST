import sys
import copy
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from FedoSSL.utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice, cluster_acc_w
from sklearn import metrics
import numpy as np
import os
# from utils_cluster import
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle


def train(args, model, device, train_label_loader, train_unlabel_loader, m, global_round):
    model.local_labeled_centroids.weight.data.zero_()  # model.local_labeled_centroids.weight.data: torch.Size([10, 512])
    labeled_samples_num = [0 for _ in range(10)]
    optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140, 180], gamma=0.1)

    model.train()
    bce = nn.BCELoss()
    m = min(m, 0.5)
    # m = 0
    ce = MarginLoss(m=-1*m)
    unlabel_ce = MarginLoss(m=0) #(m=-1*m)
    unlabel_loader_iter = cycle(train_unlabel_loader)
    bce_losses = AverageMeter('bce_loss', ':.4e')
    ce_losses = AverageMeter('ce_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')
    #
    np_cluster_preds = np.array([]) # cluster_preds
    np_unlabel_targets = np.array([])
    #
    for batch_idx, ((x, x2), target) in enumerate(train_label_loader):
        ## 各个类的不确定性权重（固定值）
        beta = 0.2
        Nk = [1600 * 5, 1600 * 5, 1600 * 5, 1600 * 5, 1600 * 5, 1600 * 5,    1600, 1600, 1600, 1600]
        Nmax = 1600 * 5
        p_weight = [beta**(1-Nk[i]/Nmax) for i in range(10)]
        #
        ((ux, ux2), unlabel_target) = next(unlabel_loader_iter)

        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)
        #
        labeled_len = len(target)
        # print("labeled_len: ", labeled_len)

        x, x2, target = x.to(device), x2.to(device), target.to(device)
        optimizer.zero_grad()
        output, feat = model(x) #output: [batch size, 10]; feat: [batch size, 512]
        output2, feat2 = model(x2)
        prob = F.softmax(output, dim=1)
        reg_prob = F.softmax(output[labeled_len:], dim=1) # unlabel data's prob
        prob2 = F.softmax(output2, dim=1)
        reg_prob2 = F.softmax(output2[labeled_len:], dim=1)  # unlabel data's prob

        # update local_labeled_centroids
        for feature, true_label in zip(feat[:labeled_len].detach().clone(), target):
            labeled_samples_num[true_label] += 1
            model.local_labeled_centroids.weight.data[true_label] += feature
        # print("before model.local_labeled_centroids.weight.data: ", model.local_labeled_centroids.weight.data)
        for idx, (feature_centroid, num) in enumerate(zip(model.local_labeled_centroids.weight.data, labeled_samples_num)):
            if num > 0:
                model.local_labeled_centroids.weight.data[idx] = feature_centroid/num
        # print("model.local_labeled_centroids.weight.data size: ", model.local_labeled_centroids.weight.data.size())
        # print("model.local_labeled_centroids.weight.data: ", model.local_labeled_centroids.weight.data)
        # print("labeled_samples_num: ", labeled_samples_num)
        # L_reg:  reg_prob中每一行的预测label
        copy_reg_prob1 = copy.deepcopy(reg_prob.detach())
        copy_reg_prob2 = copy.deepcopy(reg_prob2.detach())
        reg_label1 = np.argmax(copy_reg_prob1.cpu().numpy(), axis=1)
        reg_label2 = np.argmax(copy_reg_prob2.cpu().numpy(), axis=1)
        ### 制作target, target 除了 label=1 外与 reg_prob 一致
        for idx, (label, oprob) in enumerate(zip(reg_label1, copy_reg_prob1)):
            copy_reg_prob1[idx][label] = 1
        for idx, (label, oprob) in enumerate(zip(reg_label2, copy_reg_prob2)):
            copy_reg_prob2[idx][label] = 1
        #
        L1_loss = nn.L1Loss()
        L_reg1 = 0.0
        L_reg2 = 0.0
        for idx, (ooutput, otarget, label) in enumerate(zip(reg_prob, copy_reg_prob1, reg_label1)):
            L_reg1 = L_reg1 + L1_loss(reg_prob[idx], copy_reg_prob1[idx]) * p_weight[label]
        for idx, (ooutput, otarget, label) in enumerate(zip(reg_prob2, copy_reg_prob2, reg_label2)):
            L_reg2 = L_reg2 + L1_loss(reg_prob2[idx], copy_reg_prob2[idx]) * p_weight[label]
        L_reg1 = L_reg1 / len(reg_label1)
        L_reg2 = L_reg2 / len(reg_label2)
        #### Ours loss end
        ## 欧氏距离 ########################################################################
        # C = model.centroids.weight.data.detach().clone()
        # Z1 = feat.detach()
        # Z2 = feat2.detach()
        # cP1 = euclidean_dist(Z1, C)
        # cZ2 = euclidean_dist(Z2, C)
        ## Cluster loss begin (Orchestra) # cos-similarity ###############################
        C = model.centroids.weight.data.detach().clone().T
        Z1 = F.normalize(feat, dim=1)
        Z2 = F.normalize(feat2, dim=1)
        cP1 = Z1 @ C
        cZ2 = Z2 @ C
        ##
        tP1 = F.softmax(cP1 / model.T, dim=1)
        confidence_cluster_pred, cluster_pred = tP1.max(1) # cluster_pred: [512]; target: [170]
        tP2 = F.softmax(cZ2 / model.T, dim=1)
        #logpZ2 = torch.log(F.softmax(cZ2 / model.T, dim=1))
        # Clustering loss
        #L_cluster = - torch.sum(tP1 * logpZ2, dim=1).mean()
        # print("L_cluster: ", L_cluster)
        ## Cluster loss end (Orchestra)
        ### 统计 cluster_pred (伪标签，cluster id) 置信度 ###
        confidence_list = [0 for _ in range(10)]
        num_of_cluster = [0 for _ in range(10)]
        #mask_tmp = np.array([])
        for confidence, cluster_id in zip(confidence_cluster_pred[labeled_len:], cluster_pred[labeled_len:]):
            confidence_list[cluster_id] = confidence_list[cluster_id] + confidence
            num_of_cluster[cluster_id] = num_of_cluster[cluster_id] + 1
        for cluster_id, (sum_confidence, num) in enumerate(zip(confidence_list, num_of_cluster)):
            if num > 0:
                confidence_list[cluster_id] = confidence_list[cluster_id].cpu().detach().numpy()/num
                confidence_list[cluster_id] = np.around(confidence_list[cluster_id], 4) #保留小数点后4位
        #mask_tmp = np.append(mask_tmp, confidence_cluster_pred[labeled_len:].cpu().detach().numpy())
        threshold = 0.95
        # confidence_mask = mask_tmp > threshold
        confidence_mask = (confidence_cluster_pred[labeled_len:] > threshold)
        confidence_mask = torch.nonzero(confidence_mask)
        confidence_mask = torch.squeeze(confidence_mask)

        #print("confidence_mask: ", confidence_mask)
        #sys.exit(0)
        #if (args.epochs * global_round + epoch) % 5 == 0:

        # calculate distance
        feat_detach = feat.detach()
        feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
        cosine_dist = torch.mm(feat_norm, feat_norm.t())
        labeled_len = len(target)

        pos_pairs = []
        target_np = target.cpu().numpy()

        # label part
        for i in range(labeled_len):
            target_i = target_np[i]
            idxs = np.where(target_np == target_i)[0]
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
            else:
                selec_idx = np.random.choice(idxs, 1)
                while selec_idx == i:
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))

        # unlabel part
        unlabel_cosine_dist = cosine_dist[labeled_len:, :]
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)

        # print(pos_idx.size())
        # print(pos_idx)
        pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
        pos_pairs.extend(pos_idx) #pos_pairs size: [512,1]

        # bce + L_cluster
        cluster_pos_prob = tP2[pos_pairs, :] #cluster_pos_prob size: [512,10]
        # bce
        # cluster_pos_sim = torch.bmm(tP1.view(args.batch_size, 1, -1), cluster_pos_prob.view(args.batch_size, -1, 1)).squeeze()
        # cluster_ones = torch.ones_like(cluster_pos_sim)
        # cluster_bce_loss = bce(cluster_pos_sim, cluster_ones)
        # cross-entropy
        logcluster_pos_prob = torch.log(cluster_pos_prob)
        L_cluster = - torch.sum(tP1 * logcluster_pos_prob, dim=1).mean() #[170(label)/512-170(unlabel)]
        #
        pos_prob = prob2[pos_pairs, :]
        pos_sim = torch.bmm(prob.view(args.batch_size, 1, -1), pos_prob.view(args.batch_size, -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        print("pos:",pos_sim)
        bce_loss = bce(pos_sim, ones)
        ce_loss = ce(output[:labeled_len], target)
        # unlabel ce loss
        # unlabel_ce_loss = unlabel_ce(output[labeled_len:], cluster_pred[labeled_len:])
        ###
        #print("1:",output[labeled_len:].index_select(0,confidence_mask).size())
        #print("2:",cluster_pred[labeled_len:].index_select(0,confidence_mask).size())
        ####
        unlabel_ce_loss = unlabel_ce(output[labeled_len:].index_select(0,confidence_mask) , cluster_pred[labeled_len:].index_select(0,confidence_mask))
        np_cluster_preds = np.append(np_cluster_preds, cluster_pred[labeled_len:].cpu().numpy())
        np_unlabel_targets = np.append(np_unlabel_targets, unlabel_target.cpu().numpy())
        #
        entropy_loss = entropy(torch.mean(prob, 0))

        #loss = - entropy_loss + ce_loss + bce_loss
        # loss = ce_loss
        if global_round > 4: #4
            if global_round > 6: #6
                loss = - entropy_loss + ce_loss + bce_loss + 0.5 * L_cluster + unlabel_ce_loss #+ 2 * L_reg1 + 2 * L_reg2  # + L_cluster # 调整L_reg倍率
            else:
                loss = - entropy_loss + ce_loss + bce_loss + 0.5 * L_cluster #+ 2 * L_reg1 + 2 * L_reg2 #+ L_cluster # 调整L_reg倍率
        else:
            loss = - entropy_loss + ce_loss + bce_loss #+ 2 * L_reg1 + 2 * L_reg2 # 调整L_reg倍率
        # print("L_reg1: ", 2 * L_reg1)
        # print("L_reg2: ", 2 * L_reg2)
        # sys.exit(0)

        bce_losses.update(bce_loss.item(), args.batch_size)
        ce_losses.update(ce_loss.item(), args.batch_size)
        entropy_losses.update(entropy_loss.item(), args.batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #if ((args.epochs * global_round + epoch) % 10 == 0) and (client_id == 0):

    #if client_id == 0:
        #unlabel_acc, w_unlabel_acc = cluster_acc_w(np.array(cluster_pred[labeled_len:].cpu().numpy()), np.array(unlabel_target.cpu().numpy()))
    np_cluster_preds = np_cluster_preds.astype(int)
    unlabel_acc, w_unlabel_acc = cluster_acc_w(np_cluster_preds, np_unlabel_targets)
        #print("unlabel target: ", unlabel_target)
        #print("unlabel cluster_pred: ", cluster_pred[labeled_len:])
    scheduler.step()

def test(args, model, labeled_num, device, test_loader, epoch, client_id, is_print=True):
    model.eval()
    preds = np.array([])
    cluster_preds = np.array([]) # cluster_preds
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        C = model.centroids.weight.data.detach().clone().T
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output, feat = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            # cluster pred
            Z1 = F.normalize(feat, dim=1)
            cP1 = Z1 @ C
            tP1 = F.softmax(cP1 / model.T, dim=1)
            _, cluster_pred = tP1.max(1) # return #1: max data    #2: max data index
            #
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            # print("preds:",preds)
            cluster_preds = np.append(cluster_preds, cluster_pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
    targets = targets.astype(int)
    preds = preds.astype(int)
    cluster_preds = cluster_preds.astype(int)

    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    ## preds <-> cluster_preds ##
    origin_preds = preds
    # preds = cluster_preds
    ## local_unseen_mask (4) ##
    local_unseen_mask_4 = targets == 4
    local_unseen_acc_4 = cluster_acc(preds[local_unseen_mask_4], targets[local_unseen_mask_4])
    ## local_unseen_mask (4) ##
    local_unseen_mask_5 = targets == 5
    local_unseen_acc_5 = cluster_acc(preds[local_unseen_mask_5], targets[local_unseen_mask_5])
    ## local_unseen_mask (4) ##
    local_unseen_mask_6 = targets == 6
    local_unseen_acc_6 = cluster_acc(preds[local_unseen_mask_6], targets[local_unseen_mask_6])
    ## local_unseen_mask (4) ##
    local_unseen_mask_7 = targets == 7
    local_unseen_acc_7 = cluster_acc(preds[local_unseen_mask_7], targets[local_unseen_mask_7])
    ## local_unseen_mask (4) ##
    local_unseen_mask_8 = targets == 8
    local_unseen_acc_8 = cluster_acc(preds[local_unseen_mask_8], targets[local_unseen_mask_8])
    ## local_unseen_mask (4) ##
    local_unseen_mask_9 = targets == 9
    local_unseen_acc_9 = cluster_acc(preds[local_unseen_mask_9], targets[local_unseen_mask_9])
    ## global_unseen_mask (5-9) ##
    global_unseen_mask = targets > labeled_num

    global_unseen_acc = cluster_acc(preds[global_unseen_mask], targets[global_unseen_mask])
    ##
    # overall_acc = cluster_acc(preds, targets)
    overall_acc, w_overall_acc = cluster_acc_w(origin_preds, targets)

    # cluster_acc
    overall_cluster_acc = cluster_acc(cluster_preds, targets)
    #
    # seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    seen_acc = accuracy(origin_preds[seen_mask], targets[seen_mask])
    #
    unseen_acc, w_unseen_acc = cluster_acc_w(preds[unseen_mask], targets[unseen_mask])

    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    if is_print:
        print('epoch {}, Client id {}, Test overall acc {:.4f}, Test overall cluster acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}, local_unseen acc {:.4f}, global_unseen acc {:.4f}'.format(epoch, client_id, overall_acc, overall_cluster_acc, seen_acc, unseen_acc, local_unseen_acc_6, global_unseen_acc))
    mean_uncert = 1 - np.mean(confs)

    return mean_uncert,overall_cluster_acc
