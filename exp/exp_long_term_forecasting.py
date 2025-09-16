from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv, visual_weights
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def enhanced_test_visualization(test_data, model, device, args, folder_path, one_pred, one_true):
    try:
        # 获取完整序列
        full_series = test_data.well_series[0]
        half_len = len(full_series) // 2
        total_len = len(full_series)
        
        # 使用更长的步长（3倍pred_len）
        step_len = min(args.pred_len * 3, 384)
        
        # 准备初始编码窗口
        ctx = full_series[max(0, half_len - args.seq_len):half_len]
        if ctx.shape[0] < args.seq_len:
            pad = np.zeros((args.seq_len - ctx.shape[0],), dtype=np.float32)
            ctx = np.concatenate([pad, ctx], axis=0)
        
        # 标准化
        ctx_scaled = (ctx - test_data.scaler.mean_[0]) / test_data.scaler.scale_[0]
        window = ctx_scaled.reshape(-1, 1).astype(np.float32)
        
        # 迭代预测
        extended_pred = []
        remain = total_len - half_len
        while remain > 0:
            current_step = min(step_len, remain)
            
            # 调整输入长度
            if window.shape[0] < args.seq_len:
                pad_len = args.seq_len - window.shape[0]
                window = np.concatenate([np.zeros((pad_len, 1)), window], axis=0)
            
            x_enc = torch.from_numpy(window[-args.seq_len:]).unsqueeze(0).float().to(device)
            x_mark_enc = torch.zeros((1, x_enc.shape[1], 3), dtype=torch.float32).to(device)
            y_mark = torch.zeros((1, args.label_len + current_step, 3), dtype=torch.float32).to(device)
            
            if args.down_sampling_layers == 0:
                dec_inp = torch.zeros((1, args.label_len + current_step, 1), dtype=torch.float32).to(device)
            else:
                dec_inp = None
            
            with torch.no_grad():
                out = model(x_enc, x_mark_enc, dec_inp, y_mark)
                if isinstance(out, tuple):
                    out = out[0]
                out = out[:, -current_step:, :]
                out_np = out.detach().cpu().numpy()[0, :, 0]
            
            # 反标准化
            out_inv = out_np * test_data.scaler.scale_[0] + test_data.scaler.mean_[0]
            extended_pred.append(out_inv)
            
            # 更新窗口
            out_scaled = out_np.reshape(-1, 1).astype(np.float32)
            window = np.concatenate([window[current_step:], out_scaled], axis=0)
            remain -= current_step
        
        extended_pred = np.concatenate(extended_pred, axis=0)
        extended_pred = extended_pred[:(total_len - half_len)]
        
        # 创建三色对比数据
        input_segment = full_series[:half_len]
        true_output_segment = full_series[half_len:half_len + len(extended_pred)]
        pred_output_segment = extended_pred
        
        # 确保长度一致
        min_len = min(len(true_output_segment), len(pred_output_segment))
        true_output_segment = true_output_segment[:min_len]
        pred_output_segment = pred_output_segment[:min_len]
        
        # 绘制三色对比图
        plt.figure(figsize=(15, 8))
        x = np.arange(total_len)
        
        # 输入段（蓝色）
        plt.plot(x[:half_len], input_segment, 'b-', linewidth=2.5, label='Input Segment (Historical)', alpha=0.8)
        
        # 真实输出段（绿色）
        plt.plot(x[half_len:half_len + len(true_output_segment)], true_output_segment, 
                'g-', linewidth=2.5, label='True Output (Ground Truth)', alpha=0.8)
        
        # 预测输出段（橙色）
        plt.plot(x[half_len:half_len + len(pred_output_segment)], pred_output_segment, 
                'orange', linewidth=2.5, label='Predicted Output (Model)', alpha=0.8)
        
        # 添加分割线
        plt.axvline(x=half_len, color='red', linestyle='--', alpha=0.7, linewidth=2)
        plt.text(half_len + total_len*0.01, max(full_series)*0.9, 'Prediction Start', 
                rotation=90, fontsize=10, color='red', alpha=0.8)
        
        plt.legend(fontsize=12)
        plt.title('Well Lifecycle Prediction: Input (Blue) + True Output (Green) + Predicted Output (Orange)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps (Days)', fontsize=12)
        plt.ylabel('Gas Production', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(folder_path, 'one_well_enhanced_3color.pdf'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # 保存CSV
        df_enhanced = {
            'input_segment': np.concatenate([input_segment, np.full((total_len - half_len,), np.nan)]),
            'true_output': np.concatenate([np.full((half_len,), np.nan), 
                                         np.concatenate([true_output_segment, 
                                                        np.full((total_len - half_len - len(true_output_segment),), np.nan)])]),
            'pred_output': np.concatenate([np.full((half_len,), np.nan), 
                                         np.concatenate([pred_output_segment, 
                                                        np.full((total_len - half_len - len(pred_output_segment),), np.nan)])])
        }
        pd.DataFrame(df_enhanced).to_csv(os.path.join(folder_path, 'one_well_enhanced_3color.csv'), index=False)
        
        print(f'Enhanced visualization saved with step_len={step_len}, total_pred_len={len(extended_pred)}')
        
    except Exception as e:
        print(f'Enhanced visualization failed: {e}')


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.data == 'PEMS':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0

                pred = outputs.detach()
                true = batch_y.detach()

                # 仅对预测窗口评估
                pred = pred[:, -self.args.pred_len:, f_dim:]
                true = true[:, -self.args.pred_len:, f_dim:]

                if self.args.data == 'PEMS':
                    B, T, C = pred.shape
                    pred = pred.cpu().numpy()
                    true = true.cpu().numpy()
                    pred = vali_data.inverse_transform(pred.reshape(-1, C)).reshape(B, T, C)
                    true = vali_data.inverse_transform(true.reshape(-1, C)).reshape(B, T, C)
                    mae, mse, rmse, mape, mspe = metric(pred, true)
                    total_loss.append(mae)

                else:
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    # 对齐AMP分支：仅对预测窗口计算loss
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        checkpoints_path = './checkpoints/' + setting + '/'
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # 仅保留预测窗口
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pred_curve = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pred_curve, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

        # 保存第一个样本的曲线图与CSV（恢复到原始尺度）
        try:
            one_true = trues[0, :, -1]
            one_pred = preds[0, :, -1]
            # 如果数据集实现了逆变换，则恢复到原始尺度
            if hasattr(test_data, 'inverse_transform'):
                one_true = test_data.inverse_transform(one_true)
                one_pred = test_data.inverse_transform(one_pred)
            save_to_csv(one_true, one_pred, os.path.join(folder_path, 'one_well_true_pred.csv'))
            visual(one_true, one_pred, os.path.join(folder_path, 'one_well_true_pred.pdf'))

            # 额外绘制整口井生命周期：全真值 vs 组合曲线(前半段真值 + 后半段预测)
            # 从数据集中取到该测试样本对应的完整序列
            full_series = test_data.well_series[0]
            half_len = len(full_series) // 2
            # 注意：well_series 保持原始尺度，这里不需要逆变换
            full_series_inv = full_series
            # 组合：预测仅从后半段起点开始；此前用NaN占位，保证与全长对齐
            pred_len = one_pred.shape[0]
            pad_before = np.full((half_len,), np.nan)
            pad_after = max(0, len(full_series_inv) - (half_len + pred_len))
            pad_after = np.full((pad_after,), np.nan)
            combined = np.concatenate([pad_before, one_pred, pad_after], axis=0)
            combined = combined[:len(full_series_inv)]
            # 导出完整生命周期 CSV（便于检查拼接逻辑）
            try:
                df_full = {'true_full': full_series_inv, 'pred_combined': combined}
                pd.DataFrame(df_full).to_csv(os.path.join(folder_path, 'one_well_full_lifecycle.csv'), index=False)
            except Exception as e:
                print('Save full lifecycle CSV failed:', e)
            visual(full_series_inv, combined, os.path.join(folder_path, 'one_well_full_lifecycle.pdf'))

            # 再导出“仅预测窗口”的对比：后半段起点的真实片段 vs 模型预测
            true_pred_window = full_series_inv[half_len:half_len + pred_len]
            # 若真实长度不足pred_len，用NaN补齐以与预测对齐
            if true_pred_window.shape[0] < pred_len:
                true_pred_window = np.concatenate([
                    true_pred_window,
                    np.full((pred_len - true_pred_window.shape[0],), np.nan)
                ], axis=0)
            save_to_csv(true_pred_window, one_pred, os.path.join(folder_path, 'one_well_pred_window_true_vs_pred.csv'))
            visual(true_pred_window, one_pred, os.path.join(folder_path, 'one_well_pred_window_true_vs_pred.pdf'))


            # 增强三色可视化（更长步长）
            try:
                enhanced_test_visualization(test_data, self.model, self.device, self.args, folder_path, one_pred, one_true)
            except Exception as e:
                print(f'Enhanced 3-color visualization failed: {e}')

            # 迭代扩展预测：从中点开始，重复预测多个pred_len块直到覆盖后半段
            try:
                total_len = len(full_series_inv)
                remain = total_len - half_len
                # 准备初始编码窗口（使用原始尺度→标准化到模型尺度）
                ctx = full_series_inv[max(0, half_len - self.args.seq_len):half_len]
                if ctx.shape[0] < self.args.seq_len:
                    pad = np.zeros((self.args.seq_len - ctx.shape[0],), dtype=np.float32)
                    ctx = np.concatenate([pad, ctx], axis=0)
                # 标准化
                ctx_scaled = (ctx - test_data.scaler.mean_[0]) / test_data.scaler.scale_[0]
                window = ctx_scaled.reshape(-1, 1).astype(np.float32)  # (seq_len,1)
                extended_pred = []
                while remain > 0:
                    x_enc = torch.from_numpy(window).unsqueeze(0).float().to(self.device)  # (1,seq_len,1)
                    x_mark_enc = torch.zeros((1, x_enc.shape[1], 3), dtype=torch.float32).to(self.device)
                    y_mark = torch.zeros((1, self.args.label_len + self.args.pred_len, 3), dtype=torch.float32).to(self.device)
                    if self.args.down_sampling_layers == 0:
                        dec_inp = torch.zeros((1, self.args.label_len + self.args.pred_len, 1), dtype=torch.float32).to(self.device)
                    else:
                        dec_inp = None
                    with torch.no_grad():
                        out = self.model(x_enc, x_mark_enc, dec_inp, y_mark)
                        if isinstance(out, tuple):
                            out = out[0]
                        # 取预测窗口
                        out = out[:, -self.args.pred_len:, :]
                        out_np = out.detach().cpu().numpy()[0, :, 0]
                    # 反标准化至原始尺度
                    out_inv = out_np * test_data.scaler.scale_[0] + test_data.scaler.mean_[0]
                    extended_pred.append(out_inv)
                    # 更新窗口（使用模型尺度）
                    out_scaled = out_np.reshape(-1, 1).astype(np.float32)
                    window = np.concatenate([window[self.args.pred_len:], out_scaled], axis=0)
                    remain -= self.args.pred_len
                extended_pred = np.concatenate(extended_pred, axis=0)
                # 截断到后半段长度
                extended_pred = extended_pred[:(total_len - half_len)]
                # 组装对齐到全长：前半NaN + 扩展预测 + 末尾不足NaN
                pad_after = max(0, total_len - (half_len + extended_pred.shape[0]))
                extended_curve = np.concatenate([
                    np.full((half_len,), np.nan),
                    extended_pred,
                    np.full((pad_after,), np.nan)
                ], axis=0)
                # 导出扩展预测CSV与图
                try:
                    df_ext = {'true_full': full_series_inv, 'pred_extended': extended_curve}
                    pd.DataFrame(df_ext).to_csv(os.path.join(folder_path, 'one_well_full_lifecycle_extended.csv'), index=False)
                except Exception as e:
                    print('Save extended lifecycle CSV failed:', e)
                visual(full_series_inv, extended_curve, os.path.join(folder_path, 'one_well_full_lifecycle_extended.pdf'))
            except Exception as e:
                print('Extended multi-chunk forecast failed:', e)
        except Exception as e:
            print('Save sample CSV/plot failed:', e)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        return
