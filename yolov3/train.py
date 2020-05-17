from __future__ import division

from models import *
from my_utils.utils import *
from my_utils.datasets import *
from my_utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import importlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/garbage.py", help="path to data config file")
    args = parser.parse_args()
    cfg = importlib.import_module(args.config[:-3].replace('/', '.'))
    print(cfg)

    if not os.path.exists(cfg.work_dirs):
        os.makedirs(cfg.work_dirs)

    # logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = [s.strip() for s in list(pd.read_csv(cfg.label_list, header=None).iloc[:, 0])]

    # Initiate model
    model = Darknet(cfg.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if cfg.pretrained_weights:
        if cfg.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(cfg.pretrained_weights))
        else:
            model.load_darknet_weights(cfg.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(cfg.train, augment=True, multiscale=cfg.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    # store the name of model with best mAP
    model_best = {'mAP': 0, 'name': ''}

    for epoch in range(cfg.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % cfg.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, cfg.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                '''
                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)
                '''

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % cfg.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=cfg.valid,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=cfg.img_size,
                batch_size=1,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            # logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

            if AP.mean() > model_best['mAP']:
                temp_model_name = "ckpt_{:03d}_AP_{:.2f}.pth".format(epoch, 100 * AP.mean())
                ckpt_name = os.path.join(cfg.work_dirs, temp_model_name)
                torch.save(model.state_dict(), ckpt_name)
                model_best['mAP'] = AP.mean()
                model_best['name'] = ckpt_name

        if epoch % cfg.checkpoint_interval == 0:
            temp_model_name = "ckpt_{:03d}.pth".format(epoch)
            ckpt_name = os.path.join(cfg.work_dirs, temp_model_name)
            torch.save(model.state_dict(), ckpt_name)

        scheduler.step(epoch)
        print('The current learning rate is: ', scheduler.get_lr()[0])

    gen_model_dir(cfg, model_best['name'])
