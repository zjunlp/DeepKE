import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics


def train(epoch, train_iter, valid_iter, model, config, device):
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()

    for idx, batch in enumerate(train_iter):
        (x, lengths), y = batch.sentence, batch.relation
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        y_pred = model(x, lengths)  # batch * seq_len(max) * class

        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('epoch {}: [{} / {}] - loss: {:.6f}'.format(
            epoch, idx + 1, len(train_iter), loss.item()))

        # if idx / config.save_step == 0:
        #     save(model, epoch, idx, config)

    # eval P, R, F1
    model.eval()
    y_total = []
    y_pred_total = []
    for i, batch in enumerate(valid_iter):
        (x, lengths), y = batch.sentence, batch.relation
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        with torch.no_grad():
            y_pred = model(x, lengths)
            _, y_pred = torch.max(y_pred, 1)

        y_total.extend(y.tolist())
        y_pred_total.extend(y_pred.tolist())

    print('=' * 30)
    # use sklearn.metrics
    p_macro = metrics.precision_score(y_total, y_pred_total, average='macro')
    r_macro = metrics.recall_score(y_total, y_pred_total, average='macro')
    f1_marco = metrics.f1_score(y_total, y_pred_total, average='macro')
    print('[macro] p: {:.2f}, r: {:.2f}, f1: {:.2f}'.format(p_macro, r_macro, f1_marco))

    p_micro = metrics.precision_score(y_total, y_pred_total, average='micro')
    r_micro = metrics.recall_score(y_total, y_pred_total, average='micro')
    f1_micro = metrics.f1_score(y_total, y_pred_total, average='micro')
    print('[micro] p: {:.2f}, r: {:.2f}, f1: {:.2f}'.format(p_micro, r_micro, f1_micro))

    # print('混淆矩阵输出:\n', metrics.confusion_matrix(y, y_pred))  # 混淆矩阵输出
    # print('分类报告:\n', metrics.classification_report(y, y_pred))  # 分类报告输出

    return f1_marco, f1_micro


def save(model, epoch, step, config):
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    fp = '{}_epoch_{}_step_{}.pt'.format(model.name, epoch, step)
    fp = os.path.join(config.save_path, fp)
    torch.save(model.state_dict(), fp)
