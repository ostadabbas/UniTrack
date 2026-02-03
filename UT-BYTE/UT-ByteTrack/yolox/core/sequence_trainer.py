#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Sequence-aware trainer for integrating ByteTracker into training loop
"""

import datetime
import os
import time
from collections import defaultdict, deque
from loguru import logger

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.utils import (
    ModelEMA,
    MeterBuffer,
    WandbLogger,
    adjust_status,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupancy_for_ckpt,
    save_checkpoint,
    setup_logger,
    synchronize,
    postprocess,
)

from yolox.tracker.byte_tracker import BYTETracker
import numpy as np


class SequenceTrainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = args.local_rank
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )
        
        # Sequence tracking attributes
        self.sequence_length = getattr(exp, 'sequence_length', 8)  # frames per sequence
        self.track_buffer_size = getattr(exp, 'track_buffer_size', 30)
        self.trackers = {}  # Store trackers for each sequence
        self.sequence_cache = {}  # Store cached detections for sequences
        
        # Initialize tracking args (compatible with ByteTracker)
        self.tracking_args = type('Args', (), {
            'track_thresh': 0.6,
            'track_buffer': 30,
            'match_thresh': 0.8,
            'mot20': False,
            'frame_rate': 30
        })()

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        """
        Modified training iteration that processes sequences instead of individual frames
        """
        iter_start_time = time.time()

        # Get sequence data instead of individual frames
        sequence_data = self.get_sequence_batch()
        
        # Process sequence with tracking
        total_loss = 0
        all_losses = defaultdict(float)
        
        for seq_idx, (imgs, targets_list, img_infos) in enumerate(sequence_data):
            # Process sequence frames
            sequence_loss, losses = self.process_sequence(imgs, targets_list, img_infos, seq_idx)
            total_loss += sequence_loss
            
            # Accumulate losses for logging
            for loss_name, loss_value in losses.items():
                all_losses[loss_name] += loss_value

        # Average losses across sequences
        num_sequences = len(sequence_data)
        if num_sequences > 0:
            total_loss = total_loss / num_sequences
            for loss_name in all_losses:
                all_losses[loss_name] = all_losses[loss_name] / num_sequences

        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        
        # Update meters
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=0,  # Will be updated in get_sequence_batch
            lr=lr,
            total_loss=total_loss,
            **all_losses,
        )

    def process_sequence(self, imgs, targets_list, img_infos, seq_idx):
        """
        Process a sequence of frames with tracking
        
        Args:
            imgs: List of image tensors [frame1, frame2, ...]
            targets_list: List of target tensors [targets1, targets2, ...]  
            img_infos: List of image info tuples
            seq_idx: Sequence index for tracking
            
        Returns:
            sequence_loss: Combined loss for the sequence
            losses: Dictionary of individual loss components
        """
        # Get or create tracker for this sequence
        if seq_idx not in self.trackers:
            self.trackers[seq_idx] = BYTETracker(self.tracking_args)
        
        tracker = self.trackers[seq_idx]
        
        sequence_loss = 0
        losses = defaultdict(float)
        
        # Process each frame in the sequence
        for frame_idx, (img, targets, img_info) in enumerate(zip(imgs, targets_list, img_infos)):
            # Forward pass through model
            img = img.to(self.data_type).to(self.device)
            targets = targets.to(self.data_type).to(self.device)
            targets.requires_grad = False
            
            with torch.cuda.amp.autocast(enabled=self.amp_training):
                # Get detection outputs
                outputs = self.model(img.unsqueeze(0), targets.unsqueeze(0))
                
                # Extract detection outputs for tracking
                detection_outputs = self.extract_detections(outputs, img_info)
                
                # Update tracker with detections
                if detection_outputs is not None:
                    track_outputs = tracker.update(detection_outputs, img_info, self.input_size)
                    
                    # Update targets with track IDs from tracker
                    updated_targets = self.update_targets_with_track_ids(
                        targets, track_outputs, detection_outputs
                    )
                    
                    # Compute loss with updated track IDs
                    if self.model.training:
                        # Re-compute outputs with updated track IDs for Unitrack loss
                        loss_outputs = self.model(img.unsqueeze(0), updated_targets.unsqueeze(0))
                        frame_loss = loss_outputs["total_loss"]
                        
                        # Accumulate losses
                        sequence_loss += frame_loss
                        for loss_name, loss_value in loss_outputs.items():
                            if loss_name != "total_loss":
                                losses[loss_name] += loss_value.item() if hasattr(loss_value, 'item') else loss_value
                else:
                    # No detections, use original targets
                    if self.model.training:
                        loss_outputs = self.model(img.unsqueeze(0), targets.unsqueeze(0))
                        frame_loss = loss_outputs["total_loss"]
                        sequence_loss += frame_loss
                        
                        for loss_name, loss_value in loss_outputs.items():
                            if loss_name != "total_loss":
                                losses[loss_name] += loss_value.item() if hasattr(loss_value, 'item') else loss_value

        return sequence_loss, losses

    def extract_detections(self, model_outputs, img_info):
        """
        Extract detection outputs in format compatible with ByteTracker
        
        Args:
            model_outputs: Model prediction outputs
            img_info: Image information tuple (height, width)
            
        Returns:
            detections: numpy array of detections [x1, y1, x2, y2, score, class]
        """
        if not isinstance(model_outputs, dict) or 'total_loss' not in model_outputs:
            return None
            
        # We need to run the model in inference mode to get predictions
        # Since we're in training mode, we need to temporarily switch to eval mode
        self.model.eval()
        
        try:
            # Get the last input that was processed
            # This is a simplified approach - in practice you'd need to store the input
            # For now, we'll skip actual detection extraction and return empty
            
            # The proper implementation would be:
            # 1. Run model in inference mode with the same input
            # 2. Use postprocess() to get final detections
            # 3. Convert to ByteTracker format
            
            # For now, return empty detections to avoid errors
            return np.empty((0, 6))
            
        finally:
            self.model.train()  # Switch back to training mode

    def update_targets_with_track_ids(self, targets, track_outputs, detection_outputs):
        """
        Update training targets with track IDs from ByteTracker
        
        Args:
            targets: Original training targets
            track_outputs: Tracking results from ByteTracker
            detection_outputs: Detection outputs used for tracking
            
        Returns:
            updated_targets: Targets with updated track IDs
        """
        updated_targets = targets.clone()
        
        # Extract track IDs from tracker outputs
        if track_outputs and len(track_outputs) > 0:
            # Create mapping from detection indices to track IDs
            track_id_map = {}
            for track in track_outputs:
                track_id_map[track.track_id] = track.track_id
            
            # Update targets with track IDs
            # This is a simplified version - in practice you'd need proper association
            for i in range(updated_targets.shape[0]):
                if i < len(track_outputs):
                    updated_targets[i, 5] = track_outputs[i].track_id
        
        return updated_targets

    def get_sequence_batch(self):
        """
        Get a batch of sequences instead of individual frames
        
        Returns:
            sequence_data: List of (imgs, targets_list, img_infos) tuples
        """
        # This is a simplified implementation
        # In practice, you'd need to modify the data loader to provide sequences
        
        # For now, simulate sequence data from individual frames
        inps, targets = self.prefetcher.next()
        
        # Split batch into sequences (simplified)
        batch_size = inps.shape[0]
        sequence_data = []
        
        for i in range(0, batch_size, self.sequence_length):
            end_idx = min(i + self.sequence_length, batch_size)
            
            # Create sequence from batch
            seq_imgs = [inps[j] for j in range(i, end_idx)]
            seq_targets = [targets[j] for j in range(i, end_idx)]
            seq_img_infos = [(640, 640)] * (end_idx - i)  # Placeholder
            
            sequence_data.append((seq_imgs, seq_targets, seq_img_infos))
        
        return sequence_data

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupancy_for_ckpt(model)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(
                self.best_ap * 100
            )
        )

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
            )
            self.meter.clear_meters()

        if (self.iter + 1) % self.exp.eval_interval == 0:
            self.evaluate_and_save_model()

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def save_ckpt(self, ckpt_name):
        if self.rank == 0:
            save_checkpoint(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.epoch + 1,
                self.file_name,
                ckpt_name,
                self.use_model_ema,
                self.ema_model,
            )

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        ap50_95, ap50, summary = self.evaluator.evaluate(
            evalmodel, self.is_distributed, args.fp16
        )
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch")
        if ap50_95 > self.best_ap:
            self.best_ap = ap50_95
            self.save_ckpt("best_ckpt")

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter


def all_reduce_norm(model):
    """
    All reduce norm for distributed training
    """
    if get_world_size() == 1:
        return
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= get_world_size() 