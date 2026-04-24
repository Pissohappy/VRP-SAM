r""" Visual Prompt Encoder training (validation) code with CLIP-based text prompt """
import os
import argparse

import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist

from model.VRP_encoder_gated_fusion import VRP_encoder
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM2pred import SAM_pred


def train(args, epoch, model, sam_model, dataloader, optimizer, scheduler, training):
    r""" Train VRP_encoder model """

    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)

        protos, _ = model(
            args.condition,
            batch['query_img'],
            batch['support_imgs'].squeeze(1),
            batch['support_masks'].squeeze(1),
            batch['class_id'],
            training
        )

        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos)
        logit_mask = low_masks

        pred_mask = torch.sigmoid(logit_mask) > 0.5
        pred_mask = pred_mask.float()

        loss = model.module.compute_objective(logit_mask, batch['query_mask'])

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/home/user/zhangdantong/Datasets_HSN')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=2)  # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--condition', type=str, default='scribble', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during pascal training')
    parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])

    # CLIP / text prompt settings
    parser.add_argument('--freeze_clip', action='store_true', default=True,
                        help='Freeze CLIP text encoder parameters')
    parser.add_argument('--text_prompt_template', type=str, default='a photo of a {class_name}',
                        help='Prompt template for CLIP text encoder')

    args = parser.parse_args()

    # Distributed setting
    local_rank = args.local_rank
    dist.init_process_group(backend='nccl')
    print('local_rank: ', local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    if utils.is_main_process():
        Logger.initialize(args, training=True)
    utils.fix_randseed(args.seed)

    # Model initialization
    model = VRP_encoder(args, args.backbone, False)
    if utils.is_main_process():
        Logger.log_params(model)

    sam_model = SAM_pred()
    sam_model.to(device)

    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Device setup
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True
    )

    # Freeze backbone
    for param in model.module.layer0.parameters():
        param.requires_grad = False
    for param in model.module.layer1.parameters():
        param.requires_grad = False
    for param in model.module.layer2.parameters():
        param.requires_grad = False
    for param in model.module.layer3.parameters():
        param.requires_grad = False
    for param in model.module.layer4.parameters():
        param.requires_grad = False

    # Freeze CLIP text encoder by default
    if hasattr(model.module, 'clip_model') and args.freeze_clip:
        for param in model.module.clip_model.parameters():
            param.requires_grad = False

    # Optimizer
    trainable_params = [
        {'params': model.module.transformer_decoder.parameters()},
        {'params': model.module.downsample_query.parameters(), 'lr': args.lr},
        {'params': model.module.merge_1.parameters(), 'lr': args.lr},
        {'params': model.module.text_proj.parameters(), 'lr': args.lr},
        {'params': model.module.gate_fc.parameters(), 'lr': args.lr},
    ]

    # Optional: if later you want to finetune CLIP text encoder, turn off freeze_clip
    if hasattr(model.module, 'clip_model') and (not args.freeze_clip):
        clip_params = [p for p in model.module.clip_model.parameters() if p.requires_grad]
        if len(clip_params) > 0:
            trainable_params.append({'params': clip_params, 'lr': args.lr * 0.1})

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    Evaluator.initialize(args)

    # Dataset initialization
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(dataloader_trn)
    )

    # Training
    best_val_miou = float('-inf')
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        trn_loss, trn_miou, trn_fb_iou = train(
            args, epoch, model, sam_model, dataloader_trn, optimizer, scheduler, training=True
        )

        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(
                args, epoch, model, sam_model, dataloader_val, optimizer, scheduler, training=False
            )

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            if utils.is_main_process():
                Logger.save_model_miou(model, epoch, val_miou)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if utils.is_main_process():
            Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()

            print(
                f"[Epoch {epoch:03d}] "
                f"trn_loss={trn_loss:.4f}, val_loss={val_loss:.4f}, "
                f"trn_miou={trn_miou:.4f}, val_miou={val_miou:.4f}, "
                f"best_val_miou={best_val_miou:.4f}"
            )

    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')