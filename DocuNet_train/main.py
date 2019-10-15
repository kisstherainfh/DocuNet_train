import argparse
import os
import logging
from model import *
from dataset import *
from torch.utils.data import DataLoader
from transform import *
import torch
import time
import torch.optim as optim


def reload_model(model, logger, path=""):
    if not bool(path):
        logger.info('train from scratch')
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('*** model has been successfully loaded! ***')
        return model


def train_test(model, test_only, epoch, train_loader, test_loader, optimizer,  train_set, test_set, logger, save_path):

    # train!
    if not test_only:
        model = model.train()
        running_all, running_loss = 0., 0.
        for batch_idx, (inputs, targets_x, targets_y) in enumerate(train_loader):
            # 标准化
            inputs = Normalize(inputs)

            # to gpu
            inputs, targets_x, targets_y = inputs.cuda(), targets_x.cuda(), targets_y.cuda()

            # 过网络
            outputs = model(inputs)  # batch*2*707*500

            # loss 计算
            dec_x, dec_y = targets_x == 0, targets_y == 0
            dec_back = ((dec_x + dec_y) != 0).float()  # 背景点的logic=1
            dec_fore = 1 - dec_back  # 前景点的logic=1

            loss1_background_x = torch.max(torch.zeros(targets_x.shape), outputs[:, 0, :, :].squeeze(dim=1))
            loss1_background_y = torch.max(torch.zeros(targets_x.shape), outputs[:, 1, :, :].squeeze(dim=1))
            loss1_background = (loss1_background_x * dec_back + loss1_background_y * dec_back) / torch.sum(dec_back)

            loss2_fore = torch.sum(torch.abs(outputs - torch.stack(targets_x, targets_y).transpose(0, 1)) * dec_fore) / torch.sum(dec_fore)

            loss3_fore = torch.abs(torch.sum(outputs - torch.stack(targets_x, targets_y).transpose(0, 1)) * dec_fore) / torch.sum(dec_fore)

            loss = loss2_fore - 0.1 * loss3_fore + loss1_background

            # 梯度回传并更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print train info
            running_all += len(inputs)  # 已经跑了的样本数量
            running_loss += loss.data.item() * inputs.size(0)  # 总损失

            if batch_idx == 0:
                since = time.time()
            elif (batch_idx+1) % 100 == 0 or (batch_idx == len(train_loader)-1):
                print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                    running_all,
                    len(train_set),
                    100. * (batch_idx+1) / (len(train_loader)),  # 进度
                    running_loss / running_all,  # 每个样本损失
                    time.time()-since,  # 到目前为止花的时间
                    (time.time()-since) * (len(train_loader) / (batch_idx+1)) - (time.time()-since)))  # 还需要多少时间

            # log
            logger.info('train: Epoch:{:2}\tLoss: {:.4f}\t'.format(
                epoch,
                running_loss / len(train_set)))

            # save model
            torch.save(model.state_dict(), save_path + str(epoch + 1) + '.pt')

    # test!
    model = model.eval()
    running_all, running_loss = 0., 0.
    with torch.no_grad():
        for batch_idx, (inputs, targets_x, targets_y) in enumerate(test_loader):
            inputs, targets_x, targets_y = inputs.cuda(), targets_x.cuda(), targets_y.cuda()

            outputs = model(inputs)

            # loss 计算
            dec_x, dec_y = targets_x == 0, targets_y == 0
            dec_back = ((dec_x + dec_y) != 0).float()  # 背景点的logic=1
            dec_fore = 1 - dec_back  # 前景点的logic=1

            loss1_background_x = torch.max(torch.zeros(targets_x.shape), outputs[:, 0, :, :].squeeze(dim=1))
            loss1_background_y = torch.max(torch.zeros(targets_x.shape), outputs[:, 1, :, :].squeeze(dim=1))
            loss1_background = (loss1_background_x * dec_back + loss1_background_y * dec_back) / torch.sum(dec_back)

            loss2_fore = torch.sum(torch.abs(outputs - torch.stack(targets_x, targets_y).transpose(0, 1)) * dec_fore) / torch.sum(dec_fore)

            loss3_fore = torch.abs(torch.sum(outputs - torch.stack(targets_x, targets_y).transpose(0, 1)) * dec_fore) / torch.sum(dec_fore)

            loss = loss2_fore - 0.1 * loss3_fore + loss1_background

            # print test info
            running_all += len(inputs)  # 已经跑了的样本数量
            running_loss += loss.data.item() * inputs.size(0)  # 总损失

            if batch_idx == 0:
                since = time.time()
            elif (batch_idx + 1) % 100 == 0 or (batch_idx == len(test_loader) - 1):
                print(
                    'Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                        running_all,
                        len(train_set),
                        100. * (batch_idx + 1) / (len(test_loader)),  # 进度
                        running_loss / running_all,  # 每个样本损失
                        time.time() - since,  # 到目前为止花的时间
                        (time.time() - since) * (len(test_loader) / (batch_idx + 1)) - (
                                    time.time() - since)))  # 还需要多少时间

            # log
            logger.info('test: Epoch:{:2}\tLoss: {:.4f}\t'.format(
                epoch,
                running_loss / len(test_set)))

    return model


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0003)
    parser.add_argument('--epochs', default=60)
    parser.add_argument('--batch-size', default=8)
    parser.add_argument('--pre-trained', default=False, help='要不要用train好的模型')
    parser.add_argument('--pre-trained-model-path', default='train好的模型路径')
    args = parser.parse_args()

    # model save
    model_save_path = './model_save/'
    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)

    # log info
    filename = 'log.txt'
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # make model
    model = Net()
    model = model.cuda()

    # if use pre_trained
    if args.pre_trained:
        reload_model(model, logger, args.pre_trained_model_path)

    # criterion, optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # dataloader
    train_set = MyDataset('train')
    test_set = MyDataset('test')
    train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    # train:
    for epoch in range(args.epochs):
        model = train_test(model, args.test_only, epoch, train_loader, test_loader, optimizer,  train_set, test_set, logger, model_save_path)

        # 建议这里每过一个epoch生成几张图片看看效果


if __name__ == '__main__':
    main()
