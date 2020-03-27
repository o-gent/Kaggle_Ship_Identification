import os
import sys
from optparse import OptionParser

import numpy as np
import torch
# import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from eval import eval_net
from unet import UNet
from utils import (batch, get_ids, get_imgs_and_masks, split_ids,
                   split_train_val)


def train_net(net,
                epochs=5,
                batch_size=1,
                lr=0.1,
                val_percent=0.05,
                save_cp=True,
                gpu=False,
                img_scale=0.5):
    
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)    
    criterion = nn.BCELoss()

    dir_img = '../all/reduced/'
    dir_mask = '../all/masks/'
    dir_checkpoint = 'checkpoints/'

    # fetches ids in directory, without .jpg extension
    ids = get_ids(dir_img)
    
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    print(
    '''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

        # create a generator to try: yeild each value enumerate(batch(train, batch_size))
        # or transfer enumerate() to try loop somehow. 

        epoch_loss = 0
        exception_amount = 0

        for i, b in enumerate(batch(train, batch_size)):
            try:
                imgs = np.array([i[0] for i in b]).astype(np.float32)
                true_masks = np.array([i[1] for i in b])

                imgs = torch.from_numpy(imgs)
                true_masks = torch.from_numpy(true_masks)

                if gpu:
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()

                masks_pred = net(imgs)
                masks_probs = F.sigmoid(masks_pred)
                masks_probs_flat = masks_probs.view(-1)

                true_masks_flat = true_masks.view(-1)

                loss = criterion(masks_probs_flat, true_masks_flat)
                epoch_loss += loss.item()

                print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            except KeyboardInterrupt:
                torch.save(net.state_dict(), 'INTERRUPTED.pth')
                print('Saved interrupt')
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)

            except Exception as ah:
                print(ah)
                exception_amount += 1
                if exception_amount > 20:
                    print('things arent going good')
        
        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        val_dice = eval_net(net, val, gpu)
        print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    train_net(net=net,
                epochs=args.epochs,
                batch_size=args.batchsize,
                lr=args.lr,
                gpu=args.gpu,
                img_scale=args.scale)
