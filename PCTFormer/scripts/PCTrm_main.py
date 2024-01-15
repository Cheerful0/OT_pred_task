import os
import numpy as np
import warnings
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

from model import PCTrm_model
from data_loader import data_process
from helper import *

warnings.filterwarnings('ignore')

ITER = 5
EPOCHS = 100
LR = 1e-4
WINDOW_SIZE = 336
PATIENCE = 15
pth_path = './models/PCTrm_336sw.pth'
pic_path = '_336.png'


'''
    训练 & 验证 & 预测
'''
class PCTrm():
    def __init__(self):
        super(PCTrm, self).__init__()
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model = PCTrm_model().float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_process(flag)
        return data_set, data_loader
    
    '''
        val data
    '''
    def val(self, val_data, val_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -WINDOW_SIZE:, :]).float()
                dec_inp = torch.cat([batch_y[:, :WINDOW_SIZE, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                outputs = self.model(batch_x)
                    
                    
                f_dim = 0
                outputs = outputs[:, -WINDOW_SIZE:, f_dim:]
                batch_y = batch_y[:, -WINDOW_SIZE:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    '''
    train 
    '''
    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # pth保存路径
        path = './models'
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        # 早停
        early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

        model_optim = optim.Adam(self.model.parameters(), lr=LR)
        criterion = nn.MSELoss()

        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim, steps_per_epoch = train_steps, pct_start = 0.3, epochs = EPOCHS, max_lr = LR)

        for epoch in range(EPOCHS):
            iter_count = 0
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -WINDOW_SIZE:, :]).float()
                dec_inp = torch.cat([batch_y[:, :WINDOW_SIZE, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x)
                
                f_dim = 0
                outputs = outputs[:, -WINDOW_SIZE:, f_dim:]
                batch_y = batch_y[:, -WINDOW_SIZE:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()
                    
                adjust_learning_rate(model_optim, scheduler, epoch + 1, printout=False)
                scheduler.step()

            # loss
            train_loss = np.average(train_loss)
            val_loss = self.val(val_data, val_loader, criterion)

            print('-' * 89)
            print('| end of epoch {:3d} | train loss {:.5f} | valid loss {:.5f} '.format(epoch+1, train_loss, val_loss))
            print('-' * 89)

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # 保存模型
        best_model_path = pth_path
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    '''
        test
    '''
    def test(self, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(pth_path)))

        preds = []
        trues = []
        inputx = []
        folder_path = './res/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()

        # 记录mse、mae
        mse = [0] * ITER
        mae = [0] * ITER
        cnt = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -WINDOW_SIZE:, :]).float()
                dec_inp = torch.cat([batch_y[:, :WINDOW_SIZE, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x)

                f_dim = 0
                outputs = outputs[:, -WINDOW_SIZE:, f_dim:]
                batch_y = batch_y[:, -WINDOW_SIZE:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs 
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                # 绘制
                if i % 4 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    display_pred(gt, pd, os.path.join(folder_path, str(i) + pic_path))


                    preds = np.array(preds)
                    trues = np.array(trues)
                    inputx = np.array(inputx)

                    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
                    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
                    inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

                    # loss
                    mae[cnt] = MAE(preds, trues)
                    mse[cnt] = MSE(preds, trues)
                    print(f"cnt:{cnt}  MSE: {mse[cnt]:.5f}, MAE: {mae[cnt]:.5f}")
                    cnt += 1
                    
                    preds = []
                    trues = []
                    inputx = []
                    
        print('avg mse: ', np.mean(mse), '   std mse: ', np.std(mse))
        print('avg mae: ', np.mean(mae), '   std mae: ', np.std(mae))
        return 



if __name__=='__main__':
    model = PCTrm
    
    print('starting train...')
    pct = model()  # set experiments
    pct.train()

    print('finished\n\n')

    pct.test()
        


    