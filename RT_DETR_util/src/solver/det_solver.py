'''
by lyuwenyu
'''
import time
import json
import datetime
import csv
import os
import torch

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


def format_values_recursively(d):

    if isinstance(d, dict):
        return {k: format_values_recursively(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [format_values_recursively(v) for v in d]
    elif isinstance(d, float):
        return f"{d:.5f}"
    else:
        return d

class DetSolver(BaseSolver):

    def fit(self, ):

        print("Start training")
        self.train()
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.output_dir = os.path.join(self.cfg.output_dir, current_time)
        os.makedirs(self.output_dir, exist_ok=True)
        args = self.cfg

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        best_stat = {'epoch': -1, 'coco_eval_bbox': 0}
        start_time = time.time()

        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()

            if self.output_dir and dist.is_main_process():
                dist.save_on_master(self.state_dict(epoch), os.path.join(self.output_dir, 'last.pt'))

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            if coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
                current_coco_eval_bbox = coco_evaluator.coco_eval["bbox"].stats[0]
                if current_coco_eval_bbox > best_stat['coco_eval_bbox']:
                    best_stat['epoch'] = epoch
                    best_stat['coco_eval_bbox'] = current_coco_eval_bbox
                    dist.save_on_master(self.state_dict(epoch),  os.path.join(self.output_dir, 'best.pt'))

            # TODO
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('best_stat: ', best_stat)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            log_stats = format_values_recursively(log_stats)
            if self.output_dir and dist.is_main_process():
                csv_file_path = os.path.join(self.output_dir, "log.csv")

                if not os.path.isfile(csv_file_path):
                    with open(csv_file_path, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        headers = ['        epoch', '       train_loss', "      train_loss_vfl", '      train_loss_bbox', '     train_loss_giou', '     mAP/0.50:0.95', '       mAP/0.50', '        mAR/0.50:0.95']
                        writer.writerow(headers)

                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    row_values = [
                        '            '+ str(log_stats['epoch']),
                        '         '+ str(log_stats['train_loss']),
                        '             '+ str(log_stats['train_loss_vfl']),
                        '              '+ str(log_stats['train_loss_bbox']),
                        '             '+ str(log_stats['train_loss_giou']),
                        '           '+ str(log_stats['test_coco_eval_bbox'][0]),
                        '        '+ str(log_stats['test_coco_eval_bbox'][1]),
                        '              '+ str(log_stats['test_coco_eval_bbox'][6])
                    ]
                    writer.writerow(row_values)

                    total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)

        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
