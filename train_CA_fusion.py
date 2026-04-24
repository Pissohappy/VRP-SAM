r""" Visual Prompt Encoder training (validation) code """
import argparse
import csv
import datetime
import json
import os

import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist

from model.VRP_encoder_CA_fusion import VRP_encoder
from common.logger_CA_fusion import Logger, AverageMeter
from common.evaluation_CA_fusion import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM2pred_CA_fusion import SAM_pred


RESULT_FIELDNAMES = [
    'run_id',
    'finished_at',
    'benchmark',
    'fold',
    'backbone',
    'condition',
    'prompt_fusion',
    'clip_model',
    'epochs',
    'bsz',
    'lr',
    'seed',
    'num_query',
    'best_epoch',
    'best_val_miou',
    'best_val_fb_iou',
    'best_val_pix_acc',
    'best_val_loss',
    'final_train_loss',
    'final_train_miou',
    'final_train_fb_iou',
    'final_train_pix_acc',
    'final_val_loss',
    'final_val_miou',
    'final_val_fb_iou',
    'final_val_pix_acc',
    'logpath',
    'best_model_path',
]


def metric_to_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def metric_to_record(value, digits=5):
    value = metric_to_float(value)
    if value is None:
        return ''
    return round(value, digits)


def save_experiment_result(args, best_metrics, final_metrics):
    logpath = getattr(Logger, 'logpath', '')
    run_id = args.run_id or os.path.basename(logpath).replace('.log', '')
    row = {
        'run_id': run_id,
        'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'benchmark': args.benchmark,
        'fold': args.fold,
        'backbone': args.backbone,
        'condition': args.condition,
        'prompt_fusion': args.prompt_fusion,
        'clip_model': args.clip_model if args.prompt_fusion == 'confidence_text' else '',
        'epochs': args.epochs,
        'bsz': args.bsz,
        'lr': args.lr,
        'seed': args.seed,
        'num_query': args.num_query,
        'best_epoch': best_metrics['epoch'] if best_metrics['epoch'] is not None else '',
        'best_val_miou': metric_to_record(best_metrics['val_miou']),
        'best_val_fb_iou': metric_to_record(best_metrics['val_fb_iou']),
        'best_val_pix_acc': metric_to_record(best_metrics['val_pix_acc']),
        'best_val_loss': metric_to_record(best_metrics['val_loss']),
        'final_train_loss': metric_to_record(final_metrics['train_loss']),
        'final_train_miou': metric_to_record(final_metrics['train_miou']),
        'final_train_fb_iou': metric_to_record(final_metrics['train_fb_iou']),
        'final_train_pix_acc': metric_to_record(final_metrics['train_pix_acc']),
        'final_val_loss': metric_to_record(final_metrics['val_loss']),
        'final_val_miou': metric_to_record(final_metrics['val_miou']),
        'final_val_fb_iou': metric_to_record(final_metrics['val_fb_iou']),
        'final_val_pix_acc': metric_to_record(final_metrics['val_pix_acc']),
        'logpath': logpath,
        'best_model_path': os.path.join(logpath, 'best_model.pt') if logpath else '',
    }

    if args.result_path:
        os.makedirs(os.path.dirname(args.result_path) or '.', exist_ok=True)
        write_header = not os.path.exists(args.result_path) or os.path.getsize(args.result_path) == 0
        with open(args.result_path, 'a', newline='') as result_file:
            writer = csv.DictWriter(result_file, fieldnames=RESULT_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    if args.result_jsonl_path:
        os.makedirs(os.path.dirname(args.result_jsonl_path) or '.', exist_ok=True)
        with open(args.result_jsonl_path, 'a') as result_file:
            result_file.write(json.dumps(row) + '\n')

    Logger.info('Experiment result saved: %s\n' % (args.result_path or args.result_jsonl_path))


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
            training,
            batch['class_id'],
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

        area_inter, area_union, area_gt = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        average_meter.update(area_inter, area_union, area_gt, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    summary = average_meter.write_result('Training' if training else 'Validation', epoch)

    return summary['loss'], summary['miou'], summary['fb_iou'], summary['pix_acc']


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    default_datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Datasets_HSN')
    default_sam_checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'sam_vit_h_4b8939.pth')
    default_feature_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feats_np')
    default_result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    parser.add_argument('--datapath', type=str, default=default_datapath)
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--run_id', type=str, default='')
    parser.add_argument('--result_path', type=str, default=os.path.join(default_result_dir, 'experiment_results.csv'))
    parser.add_argument('--result_jsonl_path', type=str, default=os.path.join(default_result_dir, 'experiment_results.jsonl'))
    parser.add_argument('--bsz', type=int, default=2) # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--condition', type=str, default='scribble', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during pascal training')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)))
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--prompt_fusion', type=str, default='visual', choices=['visual', 'confidence_text'])
    parser.add_argument('--clip_model', type=str, default='ViT-B/16')
    parser.add_argument('--text_prompt_template', type=str, default='a photo of a {}.')
    parser.add_argument('--sam_checkpoint', type=str, default=os.environ.get('VRPSAM_SAM_CHECKPOINT', default_sam_checkpoint))
    parser.add_argument('--sam_feature_cache', type=str, default=os.environ.get('VRPSAM_SAM_FEATURE_CACHE', default_feature_cache))
    args = parser.parse_args()
    # Distributed setting
    local_rank = args.local_rank
    if local_rank < 0:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        args.local_rank = local_rank
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

    sam_model = SAM_pred(args.sam_checkpoint, args.sam_feature_cache, args.benchmark)
    sam_model.to(device)
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Device setup
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
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

    optimizer_params = [
        {'params': model.module.transformer_decoder.parameters()},
        {'params': model.module.downsample_query.parameters(), "lr": args.lr},
        {'params': model.module.merge_1.parameters(), "lr": args.lr},
    ]
    if args.prompt_fusion == 'confidence_text':
        optimizer_params.extend([
            {'params': model.module.text_prompt_proj.parameters(), "lr": args.lr},
            {'params': model.module.visual_confidence.parameters(), "lr": args.lr},
        ])
    optimizer = optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    Evaluator.initialize(args)

    # Dataset initialization
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')

    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs * len(dataloader_trn))
    # Training 
    best_val_miou = float('-inf')
    best_metrics = {
        'epoch': None,
        'val_loss': None,
        'val_miou': None,
        'val_fb_iou': None,
        'val_pix_acc': None,
    }
    final_metrics = {
        'train_loss': None,
        'train_miou': None,
        'train_fb_iou': None,
        'train_pix_acc': None,
        'val_loss': None,
        'val_miou': None,
        'val_fb_iou': None,
        'val_pix_acc': None,
    }
    for epoch in range(args.epochs):

        trn_loss, trn_miou, trn_fb_iou, trn_pix_acc = train(
            args, epoch, model, sam_model, dataloader_trn, optimizer, scheduler, training=True
        )
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou, val_pix_acc = train(
                args, epoch, model, sam_model, dataloader_val, optimizer, scheduler, training=False
            )

        final_metrics = {
            'train_loss': trn_loss,
            'train_miou': trn_miou,
            'train_fb_iou': trn_fb_iou,
            'train_pix_acc': trn_pix_acc,
            'val_loss': val_loss,
            'val_miou': val_miou,
            'val_fb_iou': val_fb_iou,
            'val_pix_acc': val_pix_acc,
        }

        # Save the best model
        val_miou_value = metric_to_float(val_miou)
        if val_miou_value > best_val_miou:
            best_val_miou = val_miou_value
            best_metrics = {
                'epoch': epoch,
                'val_loss': val_loss,
                'val_miou': val_miou,
                'val_fb_iou': val_fb_iou,
                'val_pix_acc': val_pix_acc,
            }
            if utils.is_main_process():
                Logger.save_model_miou(model, epoch, val_miou_value, metric_to_float(val_pix_acc))
        if utils.is_main_process():
            Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.add_scalars('data/pix_acc', {'trn_pix_acc': trn_pix_acc, 'val_pix_acc': val_pix_acc}, epoch)
            Logger.tbd_writer.flush()
    if utils.is_main_process():
        save_experiment_result(args, best_metrics, final_metrics)
        Logger.tbd_writer.close()
        Logger.info('==================== Finished Training ====================')
