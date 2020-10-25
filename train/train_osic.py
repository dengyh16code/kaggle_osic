# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import time
import argparse
import numpy as np
import os, sys, json

import torch
from torch.autograd import Variable
torch.backends.cudnn.enabled = False
from tqdm import tqdm

from models import baseline_predict_model
from data import OSICDataset,OSICDataLoader
from models import get_state





def eval(rank, args):
    print('eval start...')

    torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))

    model = baseline_predict_model()
    vqa_checkpoint = torch.load(args.eval_pt)     #load checkpoint weights
    model.load_state_dict(vqa_checkpoint['state'])   #create model
    print('--- vqa_model loaded checkpoint ---')


    eval_loader_kwargs = {
        'CT_h5': args.eval_CT_h5,
        'train_csv': args.eval_csv,
        'mode': 'eval',
        'batch_size': 1
    }



    eval_loader = EqaDataLoader(**eval_loader_kwargs)
    print('eval_loader has %d samples' % len(eval_loader.dataset))

    args.output_log_path = os.path.join(args.log_dir,
                                        'eval_' + str(rank) + '.json')

    model.eval()

    for batch in tqdm(eval_loader):
        t += 1

        model.cuda()

        CT , other_data, FVC_input = batch

        CT_var = Variable(CT.cuda())
        other_data_var = Variable(other_data.cuda())
        FVC_input_var = Variable(FVC_input.cuda())

        scores= model(CT_var, other_data_var,FVC_input_var)

        print(scores.item())



def train(rank, args):

    torch.cuda.set_device(args.gpus.index(args.gpus[rank % len(args.gpus)]))

    model = baseline_predict_model()

    lossFn = torch.nn.MSELoss().cuda()

    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate)

    train_loader_kwargs = {
        'CT_h5': args.train_CT_h5,
        'train_csv': args.train_csv,
        'mode': 'train',
        'batch_size': args.batch_size
    }

    args.output_log_path = os.path.join(args.log_dir,
                                        'train_' + str(rank) + '.json')

    train_loader = EqaDataLoader(**train_loader_kwargs)

    print('train_loader has %d samples' % len(train_loader.dataset))

    epoch = 0

    while epoch < int(args.max_epochs):
        for batch in train_loader:
            model.train()
            model.cuda()

            CT , other_data, FVC_input,FVC_label = batch

            #print('--- images dim {}'.format(images.size()))

            CT_var = Variable(CT.cuda())
            other_data_var = Variable(other_data.cuda())
            FVC_input_var = Variable(FVC_input.cuda())
            FVC_label_var = Variable(FVC_label.cuda())

            scores= model(CT_var, other_data_var,FVC_input_var)
            loss = lossFn(scores, FVC_label_var)

            # zero grad
            optim.zero_grad()

            # backprop and update
            loss.backward()
            optim.step()

        epoch += 1


        if epoch % args.save_every == 0:

            model_state = get_state(model)
            optimizer_state = optim.state_dict()

            checkpoint = {
                    'state': model_state,
                    'epoch': epoch,
                    'optimizer': optimizer_state}

            checkpoint_path = '%s/epoch_%d.pt' % (
                    args.checkpoint_dir, epoch)
            print('Saving checkpoint to %s' % checkpoint_path)
            torch.save(checkpoint, checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('-train_CT_h5', default='train.h5')
    parser.add_argument('-eval_CT_h5', default='eval.h5')
    parser.add_argument('-train_csv', default='train.csv')
    parser.add_argument('-eval_csv', default='eval.csv')
    parser.add_argument('-eval_pt', default='eval.pt')


    parser.add_argument('-mode', default='train', type=str, choices=['train','eval'])
    parser.add_argument('-batch_size', default=128, type=int)
    parser.add_argument('-learning_rate', default=1e-2, type=float)
    parser.add_argument('-max_epochs', default=10000, type=int)


    parser.add_argument('-print_every', default=10, type=int)
    parser.add_argument('-save_every', default=500, type=int)
    parser.add_argument('-identifier', default='osic')


    parser.add_argument('-checkpoint_dir', default='checkpoints/')
    parser.add_argument('-log_dir', default='logs/')
    parser.add_argument('-log', default=True, action='store_true')
    parser.add_argument('-cache', default=False, action='store_true')
    args = parser.parse_args()

    args.time_id = time.strftime("%m_%d_%H:%M")
    
    try:
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        args.gpus = [int(x) for x in args.gpus]
    except KeyError:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        args.gpus = [int(x) for x in args.gpus]
        print(args.gpus)


    args.checkpoint_dir = os.path.join(args.checkpoint_dir,
                                       args.time_id + '_' + args.identifier)
    args.log_dir = os.path.join(args.log_dir,
                                args.time_id + '_' + args.identifier)

    print(args.__dict__)

    if not os.path.exists(args.checkpoint_dir) and args.log == True:
        os.makedirs(args.checkpoint_dir)
        os.makedirs(args.log_dir)


    if args.mode == 'eval':
        eval(0, args)
    else:
        train(0, args)



