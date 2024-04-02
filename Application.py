# _*_ coding:utf-8 _*_ 
import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, default_collate
import sys

from models.PetsNet import PetsNet
from dataLoaders.OXFordData import OXFordData
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
sys.path.append('./')


# 用来去除空数据，加载图片的时候有用
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None and item[0] is not None]
    return default_collate(batch)


class App:
    def __init__(self, args):
        self.args = args
        self.prefix_dir = os.path.join("./runs", args.train_id)
        self.gpus = args.gpus
        self.device = self.find_device()
        self.model = self.get_model()
        self.Datasets = self.get_datasets(args)
        self.classfication_Loss = self.get_lossFunction()
        self.bd_loss = self.get_bd_lossFuction()
        self.opt = self.get_optim()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, "min", patience=args.lr_step,
                                                                    min_lr=0.00001)
        t = time.localtime()
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.prefix_dir, "log_tensorboard",
                                 "%s_%s_%s" % (str(t.tm_mon), str(t.tm_mday), str(t.tm_hour))))

    def get_datasets(self, args):
        train_val_set = OXFordData(args.data_path, mode='train')
        total_size = len(train_val_set)
        val_size = int(total_size * self.args.valid_percent)
        train_size = total_size - val_size

        train_set, valid_set = random_split(train_val_set, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=custom_collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=custom_collate_fn)
        return [train_loader, valid_loader]

    def get_model(self):
        if self.args.pretrain:
            model_path = os.path.join(self.prefix_dir, "model_" + str(self.args.pretrain_epoch) + ".pth")
            model = torch.load(model_path)
        else:
            if self.args.model_name == "pets":
                model = PetsNet(self.args.num_class)
            else:
                print("there is no model load, please check your code!")
        if len(self.gpus) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
        model.to(self.device)
        return model

    def find_device(self):
        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda:0" if USE_CUDA else "cpu")
        print(device)
        return device

    def get_lossFunction(self):
        loss = nn.CrossEntropyLoss()
        loss.to(self.device)
        return loss

    def get_bd_lossFuction(self):
        loss = nn.SmoothL1Loss()
        loss.to(self.device)
        return loss

    def get_optim(self):
        opt = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return opt

    def save_log(self, tags, values, n_iter):
        # dir_path = os.path.join('logs',dir_name)
        for tag, value in zip(tags, values):
            self.writer.add_scalar(tag, value, n_iter)
            self.writer.add_text(tag, str(value), n_iter)

    def write_file(self, filename, header, data):
        file_exists = os.path.isfile(filename)
        with open(filename, 'a' if file_exists else 'w') as f:
            if not file_exists or os.stat(filename).st_size == 0:
                # 文件不存在或为空，写入表头
                f.write(','.join(header) + '\n')
            # 写入数据
            f.write(','.join(map(str, data)) + '\n')

    def calculat_roc(self, net_output, label):
        # print(net_output.shape)
        y_pre = torch.argmax(net_output, dim=1)
        y_pre = y_pre.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        precision = precision_score(label, y_pre, average='macro', zero_division=0)
        recall = recall_score(label, y_pre, average='macro', zero_division=0)
        f1 = f1_score(label, y_pre, average='macro', zero_division=0)
        return f1, recall, precision

    def calculat_r2(self, predictions, targets):
        # 确保输入为cpu上的numpy数组
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        r2 = r2_score(targets, predictions)
        return r2

    def calculate_ciou(self, predictions, targets):

        # 计算预测和目标的宽度和高度
        pred_w = predictions[:, 2] - predictions[:, 0]
        pred_h = predictions[:, 3] - predictions[:, 1]
        target_w = targets[:, 2] - targets[:, 0]
        target_h = targets[:, 3] - targets[:, 1]

        # 计算预测和目标的中心点
        pred_x_center = (predictions[:, 2] + predictions[:, 0]) / 2
        pred_y_center = (predictions[:, 3] + predictions[:, 1]) / 2
        target_x_center = (targets[:, 2] + targets[:, 0]) / 2
        target_y_center = (targets[:, 3] + targets[:, 1]) / 2

        # 计算中心点之间的距离平方
        center_distance = torch.pow(pred_x_center - target_x_center, 2) + torch.pow(pred_y_center - target_y_center, 2)

        # 计算对角线长度的平方
        c_w = torch.max(predictions[:, 2], targets[:, 2]) - torch.min(predictions[:, 0], targets[:, 0])
        c_h = torch.max(predictions[:, 3], targets[:, 3]) - torch.min(predictions[:, 1], targets[:, 1])
        c_diag = torch.pow(c_w, 2) + torch.pow(c_h, 2)

        # 计算交集面积
        inter_max_xy = torch.min(predictions[:, 2:], targets[:, 2:])
        inter_min_xy = torch.max(predictions[:, :2], targets[:, :2])
        inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
        inter_area = inter[:, 0] * inter[:, 1]

        # 计算并集面积
        union_area = (pred_w * pred_h) + (target_w * target_h) - inter_area
        iou = inter_area / torch.clamp(union_area, min=1e-6)

        # 计算长宽比的一致性
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(target_w / (target_h + 1e-6)) - torch.atan(pred_w / (pred_h + 1e-6)), 2)
        with torch.no_grad():
            alpha = v / ((1 - iou) + v)
        alpha[v < 1e-6] = 0

        # 计算CIoU损失
        ciou = iou - (center_distance / c_diag + v * alpha)

        return ciou.mean()


    def update(self, data):
        img, label_bd, label_class = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
        self.opt.zero_grad()
        bbox, class_logits = self.model(img)
        class_loss = self.classfication_Loss(class_logits, label_class)
        bd_loss = self.bd_loss(bbox, label_bd)
        loss = class_loss * self.args.cls_loss_weight + bd_loss * (1-self.args.cls_loss_weight)
        loss.backward()
        self.opt.step()
        lr = self.opt.state_dict()['param_groups'][0]['lr']
        f1, recall, precision = self.calculat_roc(class_logits, label_class)
        ciou = self.calculate_ciou(bbox, label_bd)

        return [class_loss.cpu().detach().numpy(), f1, recall, precision, bd_loss.cpu().detach().numpy(), ciou.cpu().detach().numpy(), loss.cpu().detach().numpy(), lr]

    def forward(self, data):
        img, label_bd, label_class = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)

        bbox, class_logits = self.model(img)
        cls_loss = self.classfication_Loss(class_logits, label_class)
        bd_loss = self.bd_loss(bbox, label_bd)
        loss = cls_loss * self.args.cls_loss_weight + bd_loss * (1-self.args.cls_loss_weight)
        f1, recall, precision = self.calculat_roc(class_logits, label_class)
        ciou = self.calculate_ciou(bbox, label_bd)

        return [cls_loss.cpu().detach().numpy(), f1, recall, precision, bd_loss.cpu().detach().numpy(), ciou.cpu().detach().numpy(), loss.cpu().detach().numpy()]

    def save_model(self, mode="last"):
        if mode not in ["last", "best"]:
            raise ValueError("save 'mode' should be either 'last' or 'best'")
        if mode == "last":
            torch.save(self.model.state_dict(),
                       os.path.join(self.prefix_dir, "last.pt"))
        else:
            torch.save(self.model.state_dict(),
                      os.path.join(self.prefix_dir, "best.pt"))

    def validation(self, valid_loader, n_iter):
        print("start validation -->")
        val_loss, val_f1, val_recall, val_precision, val_bd_loss, val_bd_ciou, val_total_loss = 0., 0., 0., 0., 0., 0., 0.
        step = 0
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Validation")
        for v_batch_i, v_data in pbar:
            v_values = self.forward(v_data)
            val_loss += v_values[0]
            val_f1 += v_values[1]
            val_recall += v_values[2]
            val_precision += v_values[3]
            val_bd_loss += v_values[4]
            val_bd_ciou += v_values[5]
            val_total_loss += v_values[6]
            step += 1
            # 更新进度条描述
            pbar.set_postfix({
                'Cls Loss': f"{val_loss / step:.4f}",
                'Precision': f"{val_precision / step:.4f}",
                'Recall': f"{val_recall / step:.4f}",
                'F1': f"{val_f1 / step:.4f}",
                'BD Loss': f"{val_bd_loss / step:.4f}",
                'BD CIOU': f"{val_bd_ciou / step:.4f}",
                'Total Loss': f"{val_total_loss / step:.4f}"
            })

        all_rate = [val_loss/step, val_f1/step, val_recall/step, val_precision/step, val_bd_loss/step, val_bd_ciou/step, val_total_loss/step]
        self.save_log(['validation/cls_loss', 'validation/f1_score', 'validation/recall', 'validation/precision',
                       'validation/bd_loss', 'validation/bd_ciou', 'validation/total_loss'], all_rate,
                      n_iter)

        return all_rate

    def train(self):
        train_loader, valid_loader = self.Datasets
        n_iter = 0
        best_loss = 1000.
        for epoch in range(self.args.epochs):
            train_loss, train_f1, train_recall, train_precision, train_bd_loss, train_bd_ciou, total_loss = 0., 0., 0., 0., 0., 0., 0.
            step = 0

            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{self.args.epochs}")
            for batch_i, data in pbar:
                values = self.update(data)
                train_loss += values[0]
                train_f1 += values[1]
                train_recall += values[2]
                train_precision += values[3]
                train_bd_loss += values[4]
                train_bd_ciou += values[5]
                total_loss += values[6]
                step += 1
                n_iter += data[1].shape[0]
                # 更新进度条描述
                pbar.set_postfix({
                    'Cls Loss': f"{train_loss / step:.4f}",
                    'Precision': f"{train_precision / step:.4f}",
                    'Recall': f"{train_recall / step:.4f}",
                    'F1': f"{train_f1 / step:.4f}",
                    'BD Loss': f"{train_bd_loss / step:.4f}",
                    'BD CIOU': f"{train_bd_ciou / step:.4f}",
                    'Total Loss': f"{total_loss / step:.4f}"
                })

                all_rate = [train_loss / step, train_f1 / step, train_recall / step, train_precision / step,
                            train_bd_loss / step,  train_bd_ciou / step, total_loss / step, values[-1]]
                self.save_log(
                    ['train/loss_cls', 'train/f1_score', 'train/recall', 'train/precision', 'train/loss_bd', 'train/ciou', 'train/total_loss', 'train/learning_rate'],
                    all_rate, n_iter)

            print("[train] Epoch Summary: ")
            print(
                f"Classification - Loss: {train_loss / step:.4f}, Precision: {train_precision / step:.4f}, Recall: {train_recall / step:.4f}, F1: {train_f1 / step:.4f}")
            print(f"BD - Loss: {train_bd_loss / step:.4f}, Ciou: {train_bd_ciou / step:.4f}")
            print(f"Total - Loss: {total_loss / step:.4f}")
            header = ['Epoch', 'Cls_Loss', 'Precision', 'Recall', 'F1', 'BD_Loss', 'BD_ciou', 'Total_Loss', 'Learning_Rate']
            data = [epoch, train_loss / step, train_precision / step, train_recall / step, train_f1 / step,
                    train_bd_loss / step, train_bd_ciou / step, total_loss / step, values[-1]]
            self.write_file(os.path.join(self.prefix_dir, "train_epoch.txt"), header, data)

            # 每 10 epoch valid一次
            if (epoch+1) % self.args.valid_per_epoch == 0:

                valid_result = self.validation(valid_loader, n_iter)
                print("[validation] Epoch Summary: ")
                print(
                    f"Classification - Loss: {valid_result[0]:.4f}, Precision: {valid_result[3]:.4f}, Recall: {valid_result[2]:.4f}, F1: {valid_result[1]:.4f}")
                print(f"BD - Loss: {valid_result[4]:.4f}, CIOU: {valid_result[5]:.4f}")
                self.scheduler.step(valid_result[-1])
                valid_header = ['Epoch', 'Cls_Loss', 'Precision', 'Recall', 'F1', 'BD_Loss', 'BD_Ciou']
                valid_data = [epoch, valid_result[0], valid_result[3], valid_result[2], valid_result[1],
                              valid_result[4], valid_result[5]]
                self.write_file(os.path.join(self.prefix_dir, "valid_epoch.txt"), valid_header, valid_data)

            self.save_model(mode='last')
            if total_loss / n_iter < best_loss:
                self.save_model(mode='best')
