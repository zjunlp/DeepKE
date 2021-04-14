import torch
import logging
import matplotlib.pyplot as plt
from metrics import PRMetric

logger = logging.getLogger(__name__)


def train(epoch, model, dataloader, optimizer, criterion, device, writer, cfg):
    """
    训练模型
        Args:
            epoch (int): 训练步数
            model (class): 训练的模型
            dataloader (dict): 数据集
            optimizer (Callable): 优化器
            criterion (Callable): 损失函数
            device (torch.device): 训练的设备
            writer (class): 输出
            cfg: 配置文件
        Return:
            losses[-1] : loss值
    """
    model.train()

    metric = PRMetric()
    losses = []

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        for key, value in x.items():
            x[key] = value.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)

        if cfg.model_name == 'capsule':
            loss = model.loss(y_pred, y)
        else:
            loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()

        metric.update(y_true=y, y_pred=y_pred)
        losses.append(loss.item())

        data_total = len(dataloader.dataset)
        data_cal = data_total if batch_idx == len(dataloader) else batch_idx * len(y)
        if (cfg.train_log and batch_idx % cfg.log_interval == 0) or batch_idx == len(dataloader):
            # p r f1 皆为 macro，因为micro时三者相同，定义为acc
            acc, p, r, f1 = metric.compute()
            logger.info(f'Train Epoch {epoch}: [{data_cal}/{data_total} ({100. * data_cal / data_total:.0f}%)]\t'
                        f'Loss: {loss.item():.6f}')
            logger.info(f'Train Epoch {epoch}: Acc: {100. * acc:.2f}%\t'
                        f'macro metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')

    if cfg.show_plot and not cfg.only_comparison_plot:
        if cfg.plot_utils == 'matplot':
            plt.plot(losses)
            plt.title(f'epoch {epoch} train loss')
            plt.show()

        if cfg.plot_utils == 'tensorboard':
            for i in range(len(losses)):
                writer.add_scalar(f'epoch_{epoch}_training_loss', losses[i], i)

    return losses[-1]


def validate(epoch, model, dataloader, criterion, device, cfg):
    """
    验证模型
        Args:
            epoch (int): 训练步数
            model (class): 训练的模型
            dataloader (dict): 数据集
            optimizer (Callable): 优化器
            criterion (Callable): 损失函数
            device (torch.device): 训练的设备
            cfg: 配置文件
        Return:
            f1 : f1值
            loss : loss值
    """
    model.eval()

    metric = PRMetric()
    losses = []

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        for key, value in x.items():
            x[key] = value.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = model(x)

            if cfg.model_name == 'capsule':
                loss = model.loss(y_pred, y)
            else:
                loss = criterion(y_pred, y)

            metric.update(y_true=y, y_pred=y_pred)
            losses.append(loss.item())

    loss = sum(losses) / len(losses)
    acc, p, r, f1 = metric.compute()
    data_total = len(dataloader.dataset)

    if epoch >= 0:
        logger.info(f'Valid Epoch {epoch}: [{data_total}/{data_total}](100%)\t Loss: {loss:.6f}')
        logger.info(f'Valid Epoch {epoch}: Acc: {100. * acc:.2f}%\tmacro metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')
    else:
        logger.info(f'Test Data: [{data_total}/{data_total}](100%)\t Loss: {loss:.6f}')
        logger.info(f'Test Data: Acc: {100. * acc:.2f}%\tmacro metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')

    return f1, loss
