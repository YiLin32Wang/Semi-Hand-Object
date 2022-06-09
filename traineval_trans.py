import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim

from utils.utils import Monitor, get_dataset, get_network, get_network_Trans, print_args, save_args, load_checkpoint, save_checkpoint
from utils.epoch import Eval_epoch_Trans, Train_epoch, Eval_epoch, Train_epoch_Trans, Val_epoch_Trans
from utils.options import add_opts
from utils.renderer import Renderer
from utils._mano import MANO
from src.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'


def main(args):
    # Initialize randoms seeds
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    args.local_rank = 0

    # create exp result dir
    os.makedirs(args.host_folder, exist_ok=True)
    #TODO: initialize renderer for the visualization
    mano_model = MANO()
    renderer = Renderer(faces=mano_model.face)
    args.mano_model = mano_model
    #print("Init distributed training on local rank {}".format(args.local_rank))
    #torch.cuda.set_device(args.local_rank)
    #torch.distributed.init_process_group(
    #        backend='nccl', init_method='env://', world_size=1, rank=args.local_rank)
    #synchronize()
    # Initialize model
    model = get_network_Trans(args)

    if args.use_cuda and torch.cuda.is_available():
        print("Using {} GPUs !".format(torch.cuda.device_count()))
        model.cuda()

    start_epoch = 0
    if not args.evaluate:
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(model_params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150, 200], gamma=args.lr_decay_gamma)
        # TODO: train split
        train_dat = get_dataset(args, mode="train")
        print("training dataset size: {}".format(len(train_dat)))
        train_loader = torch.utils.data.DataLoader(train_dat, batch_size=args.train_batch, shuffle=True,
                                                   num_workers=int(args.workers), pin_memory=False, drop_last=False)
        # TODO: validation split
        val_dat = get_dataset(args, mode="validation")
        print(f"validation dataset size:{len(val_dat)}")
        val_loader = torch.utils.data.DataLoader(val_dat, batch_size=args.test_batch, shuffle=False, num_workers=int(args.workers), pin_memory=False, drop_last=False)

        monitor = Monitor(hosting_folder=args.host_folder)
        device = torch.device('cuda')if torch.cuda.is_available() and args.use_cuda else torch.device('cpu')
        if args.resume is not None:
            start_epoch = load_checkpoint(model, resume_path=args.resume, strict=False, device=device)

    else:
        assert args.resume is not None, "need trained model for evaluation"
        device = torch.device('cuda')if torch.cuda.is_available() and args.use_cuda else torch.device('cpu')
        start_epoch = load_checkpoint(model, resume_path=args.resume, strict=False, device=device)
        args.epochs = start_epoch + 1

    
    
    # Initialize testing dataset
    test_dat = get_dataset(args, mode="evaluation")
    print("evaluation dataset size: {}".format(len(test_dat)))
    test_loader = torch.utils.data.DataLoader(test_dat, batch_size=args.test_batch,
                                             shuffle=False, num_workers=int(args.workers),
                                             pin_memory=False, drop_last=False)

    #TODO: change 'single_epoch' function to 'Epoch'-'Train_epoch'-'Val_epoch' class, first initialize the epoch object
    if not args.evaluate:
        train_epoch = Train_epoch_Trans(dataloader=train_loader, model=model, optimizer=optimizer,save_path=args.host_folder, mode="train", save_results=False, use_cuda=args.use_cuda,  args=args, renderer=renderer)
        # TODO: Initialize validation epoch
        val_epoch = Val_epoch_Trans(dataloader=val_loader, model=model, optimizer=optimizer,save_path=args.host_folder, mode="validation", save_results=False, use_cuda=args.use_cuda,  args=args, renderer=renderer)
        
    
    test_epoch = Eval_epoch_Trans(dataloader=test_loader, model=model, 
                            optimizer=None, save_path=args.host_folder,
                            mode="evaluation", save_results=args.save_results, use_cuda=args.use_cuda,
                            indices_order=test_dat.jointsMapSimpleToMano if hasattr(test_dat, "jointsMapSimpleToMano") else None, args=args,renderer=renderer)

    xyz_dict = {"auc":[],"mean":[], "al_auc":[], "al_mean":[]}
    handmesh_dict = {"auc":[], "mean":[], "al_auc":[], "al_mean":[]}
    val_epoch_list = []
    for epoch in range(start_epoch, args.epochs):
        train_dict = {}
        # Evaluate on validation set
        #print(f"epoch:{epoch+1}; test_freq:{args.test_freq}; save_result:{args.save_results}.")

        if not args.evaluate:
            print("Using lr {}".format(optimizer.param_groups[0]["lr"]))
            # train_avg_meters = train_epoch.update(epoch=epoch)
            # train_dict = {meter_name: meter.avg
            #               for (meter_name, meter) in train_avg_meters.average_meters.items()}
            # monitor.log_train(epoch + 1, train_dict)

            if (epoch+1) % args.snapshot == 0:
                print(f"save epoch {epoch+1} checkpoint to {args.host_folder}")
                save_checkpoint(
                {
                    "epoch": epoch,
                    "network": args.network,
                    "state_dict": model.state_dict(),
                },
                checkpoint=args.host_folder, filename=f"checkpoint_{epoch+1}.pth.tar")

            if args.lr_decay_gamma:
                if args.lr_decay_step is None:
                    scheduler.step(train_dict["joints3d_loss"])
                else:
                    scheduler.step()

            #continue
            if (epoch) % args.val_freq == 0:
                val_epoch_list.append(epoch+1)
                with torch.no_grad():
                    val_avg_meters, xyz_dict, handmesh_dict  = val_epoch.update(epoch, xyz_dict, handmesh_dict, val_epoch_list)
                val_dict = {meter_name: meter.avg for (meter_name, meter) in val_avg_meters.average_meters.items()}
                monitor.log_val(epoch + 1, val_dict)


        if args.evaluate or (epoch+1) % args.test_freq == 0:
            with torch.no_grad():
                test_epoch.update(epoch=epoch if not args.evaluate else None)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand-Object training")
    add_opts(parser)
    CUDA_LAUNCH_BLOCKING=1

    args = parser.parse_args()
    #args.test_freq = 10
    args.save_results = True
    #args.snapshot = 10

    print_args(args)
    save_args(args, save_folder=args.host_folder, opt_prefix="option")
    main(args)
    print("All done !")