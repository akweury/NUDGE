# Created by jing at 06.11.23
import torch
import torch.nn as nn

def code_loss(code_output, code_gt):
    code_gt = code_gt.squeeze(0)
    loss = nn.MSELoss()
    output = loss(code_output, code_gt)
    return output



def label_loss(label_out, label_gt):
    label_gt = label_gt.squeeze(0)
    loss = nn.MSELoss()
    output = loss(label_out, label_gt)
    return output

def classfication_loss(label_out, label_gt):
    label_gt = label_gt.squeeze(0)
    loss = nn.CrossEntropyLoss()
    output = loss(label_out, label_gt)
    return output


def shape_loss(gt_shape, out_center):
    gt_order =  torch.tensor(list(range(len(out_center)))).to(out_center.device)
    _, out_order = torch.sort(out_center[:,0])
    correct = torch.sum(gt_order==out_order)

    return correct


def action_loss(action_prob, pred_action_prob):
    output = label_loss(action_prob, pred_action_prob)
    return output