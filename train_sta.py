import os
import math
import time
import yaml
import argparse
import torch
import torch.optim as optim
import os.path as op
import numpy as np
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
import utils
import utils.deep_learning as dpl
import dataset
from sta_network import Net
import copy
import time


def receive_arg():
    """Process all hyper-parameters and experiment settings.
    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='option_R3_mfqev2_1D.yml',
        help='Path to option YAML file.'
    )
    args = parser.parse_args()
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path

    opts_dict['train']['rank'] = 0
    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log.log"
    )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
    )

    opts_dict['train']['is_dist'] = False

    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
    )

    return opts_dict


def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    unit = opts_dict['train']['criterion']['unit']
    num_iter = int(opts_dict['train']['num_iter'])
    interval_print = int(opts_dict['train']['interval_print'])
    interval_val = int(opts_dict['train']['interval_val'])

    log_dir = op.join("exp", opts_dict['train']['exp_name'])
    print("log_dir", log_dir)
    if not os.path.exists(log_dir):
        utils.mkdir(log_dir)
    log_fp = open(opts_dict['train']['log_path'], 'w')

    # log all parameters
    msg = (
        f"{'<' * 10} Hello {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"\n{'<' * 10} Options {'>' * 10}\n"
        f"{utils.dict2str(opts_dict)}"
    )
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    seed = opts_dict['train']['random_seed']
    utils.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True  # speed up
    # torch.backends.cudnn.deterministic = True  # if reproduce

    # create datasets
    train_ds_type = opts_dict['dataset']['train']['type']
    radius = opts_dict['network']['radius']
    assert train_ds_type in dataset.__all__, \
        "Not implemented!"
    train_ds_cls = getattr(dataset, train_ds_type)
    train_ds = train_ds_cls(
        opts_dict=opts_dict['dataset']['train'],
        radius=radius
    )

    rank = 0
    # create datasamplers
    train_sampler = utils.DistSampler(
        dataset=train_ds,
        num_replicas=1,  # opts_dict['train']['num_gpu'],
        rank=rank,
        ratio=opts_dict['dataset']['train']['enlarge_ratio']
    )

    # create dataloaders
    train_loader = utils.create_dataloader(
        dataset=train_ds,
        opts_dict=opts_dict,
        sampler=train_sampler,
        phase='train',
        seed=opts_dict['train']['random_seed']
    )

    assert train_loader is not None

    batch_size = opts_dict['dataset']['train']['batch_size_per_gpu']
    num_iter_per_epoch = math.ceil(len(train_ds) * \
                                   opts_dict['dataset']['train']['enlarge_ratio'] / batch_size)

    print(num_iter, opts_dict['dataset']['train']['enlarge_ratio'])
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)
    print("num_epoch:", num_epoch)

    # create dataloader prefetchers
    tra_prefetcher = utils.CPUPrefetcher(train_loader)

    # ==========
    # create model    ,find_unused_parameters=True
    # ==========
    model = Net(1, 64)
    model = torch.nn.DataParallel(model)

    if False:
        # 加载预训练模型 load pre-trained generator      ,, map_location='cpu'  ,map_location={'cuda:0':'cuda:2'})
        ckp_path = '/home/shuchen/OVSCQE/stdf-pytorch-master/exp/V13Lite2SFEB_CBAM_T_22_New/ckp_265000_val.pth'
        checkpoint = torch.load(ckp_path)
        state_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        if ('module.' in list(state_dict.keys())[0]) and (
                'module.' not in list(model_dict.keys())[0]):  # multi-gpu pre-trained -> single-gpu training
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove module
                new_state_dict[name] = v
            # model.load_state_dict(new_state_dict)
            print(f'loaded from {ckp_path}')
            log_fp.write(f'loaded from {ckp_path}' + '\n')
            log_fp.flush()
        elif ('module.' not in list(state_dict.keys())[0]) and (
                'module.' in list(model_dict.keys())[0]):  # single-gpu pre-trained -> multi-gpu training
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = 'module.' + k  # add module
                new_state_dict[name] = v
            # model.load_state_dict(new_state_dict)
            print(f'loaded from2 {ckp_path}')
            log_fp.write(f'loaded from2 {ckp_path}' + '\n')
            log_fp.flush()
        else:  # the same way of training  ,strict=False
            # model.load_state_dict(state_dict)
            new_state_dict = state_dict
            print(f'loaded from3 {ckp_path}')
            log_fp.write(f'loaded from3 {ckp_path}' + '\n')
            log_fp.flush()

        pretrained_dict = new_state_dict
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    # ==========
    # define loss func & optimizer & scheduler & scheduler & criterion
    # ==========
    assert opts_dict['train']['loss'].pop('type') == 'CombinedLoss', \
        "Not implemented."
    loss_func = dpl.CombinedLoss()

    # define optimizer
    assert opts_dict['train']['optim'].pop('type') == 'Adam', \
        "Not implemented."
    optimizer = optim.Adam(
        model.parameters(),
        **opts_dict['train']['optim']
    )

    # define scheduler
    if opts_dict['train']['scheduler']['is_on']:
        assert opts_dict['train']['scheduler'].pop('type') == \
               'CosineAnnealingRestartLR', "Not implemented."
        del opts_dict['train']['scheduler']['is_on']
        scheduler = utils.CosineAnnealingRestartLR(
            optimizer,
            **opts_dict['train']['scheduler']
        )
        opts_dict['train']['scheduler']['is_on'] = True

    # define criterion
    assert opts_dict['train']['criterion'].pop('type') == \
           'PSNR', "Not implemented."
    criterion = utils.PSNR()

    start_iter = 0  # should be restored
    start_epoch = start_iter // num_iter_per_epoch

    msg = (
        f"\n{'<' * 10} Dataloader {'>' * 10}\n"
        f"total iters: [{num_iter}]\n"
        f"total epochs: [{num_epoch}]\n"
        f"iter per epoch: [{num_iter_per_epoch}]\n"
        f"start from iter: [{start_iter}]\n"
        f"start from epoch: [{start_epoch}]"
    )
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    msg = f"\n{'<' * 10} Training {'>' * 10}"
    print(msg)
    log_fp.write(msg + '\n')

    model = model.cuda()
    criterion = criterion.cuda()
    model.train()
    num_iter_accum = start_iter

    for current_epoch in range(start_epoch, num_epoch + 1):

        # fetch the first batch
        tra_prefetcher.reset()

        train_data = tra_prefetcher.next()
        while train_data is not None:
            num_iter_accum += 1
            if num_iter_accum > num_iter:
                break

            # get data
            gt_data = train_data['gt'].cuda()  # .to(rank)  # (B [RGB] H W)
            lq_data = train_data['lq'].cuda()  # .to(rank)  # (B T [RGB] H W)

            gt_f0 = gt_data[:, 2, :, :, :]
            lq_f2n = lq_data[:, 0, :, :, :]
            lq_f1n = lq_data[:, 1, :, :, :]
            lq_f0 = lq_data[:, 2, :, :, :]
            lq_f1 = lq_data[:, 3, :, :, :]
            lq_f2 = lq_data[:, 4, :, :, :]
            b, t, c, _, _ = lq_data.shape

            enhanced = model(f2n=lq_f2n, f1n=lq_f1n, f0=lq_f0, f1=lq_f1, f2=lq_f2)
            loss = loss_func(enhanced, gt_f0)
            loss2 = loss_func(lq_f0, gt_f0)

            optimizer.zero_grad()  # zero grad
            loss.backward()  # cal grad
            optimizer.step()  # update parameters

            # display & log
            lr = optimizer.param_groups[0]['lr']

            loss_item = loss.item()
            loss2_item = loss2.item()

            msg = (
                f"iter: [{num_iter_accum}]/{num_iter}, "
                f"epoch: [{current_epoch}]/{num_epoch - 1}, "
                "lr: [{:.3f}]x1e-4, loss: [{:.6f}], loss2: [{:.6f}], delta: [{:.6f}]".format(
                    lr * 1e4, loss_item, loss2_item, loss_item - loss2_item
                )
            )
            print(msg)

            if num_iter_accum % interval_print == 0:
                log_fp.write(msg + '\n')
                log_fp.flush()

            if ((num_iter_accum % (interval_val) == 0) or (num_iter_accum == 1) or (num_iter_accum == num_iter)):
                # save model
                checkpoint_save_path = (
                    f"{opts_dict['train']['checkpoint_save_path_pre']}"
                    f"{num_iter_accum}"
                    "_val.pth"
                )
                state = {
                    'num_iter_accum': num_iter_accum,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if opts_dict['train']['scheduler']['is_on']:
                    state['scheduler'] = scheduler.state_dict()
                torch.save(state, checkpoint_save_path)

                # log
                msg = (
                    "> model saved at {:s}\n"
                ).format(
                    checkpoint_save_path
                )
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()

            if num_iter_accum == 100000 or num_iter_accum == 200000:
                model_clone = copy.deepcopy(model.state_dict())
                opti_clone = copy.deepcopy(optimizer.state_dict())
                lr = optimizer.param_groups[0]['lr']
                msg = "adjusting lr from " + str(lr) + " to " + str(lr * 0.5)
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()
                model.load_state_dict(model_clone)
                optimizer.load_state_dict(opti_clone)
                optimizer.param_groups[0]["lr"] = lr * 0.5
                model.train()
            # fetch next batch
            # start_time = time.time()
            train_data = tra_prefetcher.next()
            # elapsed = time.time() - start_time
        # print('data featch time: ', elapsed)

    msg = (
        f"\n{'<' * 10} Goodbye {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]"
    )
    print(msg)
    log_fp.write(msg + '\n')

    log_fp.close()

    state = {
        'num_iter_accum': num_iter_accum,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if opts_dict['train']['scheduler']['is_on']:
        state['scheduler'] = scheduler.state_dict()
    torch.save(state, 'final_eval_32_LDP.pth')


if __name__ == '__main__':
    print("Train")
    main()
