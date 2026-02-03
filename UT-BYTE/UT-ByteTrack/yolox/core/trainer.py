#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger

import datetime
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.utils import (
    ModelEMA,
    MeterBuffer,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    load_ckpt,
    save_checkpoint,
    setup_logger,
    synchronize,
    postprocess,
)

from yolox.tracker.byte_tracker import BYTETracker
import numpy as np


def is_parallel(model):
    """Check if model is in parallel mode"""
    return hasattr(model, 'module')


class Trainer:
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
        
        # Initialize tracking for UniTrack if enabled
        self.enable_tracking = getattr(exp, 'enable_unitrack', False)
        if self.enable_tracking:
            # Initialize batch-level tracker
            self.tracker = None
            self.tracking_args = type('Args', (), {
                'track_thresh': 0.6,
                'track_buffer': 30,
                'match_thresh': 0.8,
                'mot20': False,
                'frame_rate': 30
            })()
            
            # Frame counter for tracking
            self.frame_count = 0
            self.track_id_mapping = {}  # Map ground truth to track IDs
            
        logger.info(f"UniTrack tracking enabled: {self.enable_tracking}")

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

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

            # Print all losses after each iteration
            if hasattr(self, 'loss'):
                print(f"[Iter {self.iter+1}/{self.max_iter}] loss: {self.loss:.4f}")
            if hasattr(self, 'loss_unitrack'):
                print(f"[Iter {self.iter+1}/{self.max_iter}] unitrack_loss: {self.loss_unitrack:.4f}")
            # Optionally, print more detailed losses if available
            if hasattr(self, 'loss_iou') and hasattr(self, 'loss_obj') and hasattr(self, 'loss_cls'):
                print(f"[Iter {self.iter+1}/{self.max_iter}] iou: {self.loss_iou:.4f}, obj: {self.loss_obj:.4f}, cls: {self.loss_cls:.4f}")

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        
        # Check if the model uses UniTrack
        if hasattr(self.model, 'head') and hasattr(self.model.head, 'enable_unitrack') and self.model.head.enable_unitrack:
            # Process with tracking for UniTrack - KEEP track_ids column for Unitrack loss
            targets = self.process_batch_with_tracking(inps, targets)
            # targets should maintain shape [B, max_labels, 6] with track_ids in column 5
        else:
            # Original behavior - extract track_ids but only use first 5 columns
            track_ids = targets[:, :, 5]
            targets = targets[:, :, :5]
        
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)
        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )
        
        # Detailed loss logging every iteration for first 100 iterations, then every 10 iterations
        if self.iter < 100 or self.iter % 10 == 0:
            self.log_detailed_losses(outputs)
        
        # Visualization every 50 iterations - fix the condition
        if self.iter % 50 == 0:
            if self.enable_tracking:
                logger.info(f"Triggering visualization at iter {self.iter}")
                self.visualize_tracking_results(inps, targets, outputs)
            else:
                logger.info(f"Visualization skipped at iter {self.iter} - tracking not enabled")
        
        # Debug logging for visualization condition
        if self.iter in [1, 2, 3, 49, 50, 51, 99, 100, 101]:
            logger.info(f"DEBUG - Iter {self.iter}: visualization condition {self.iter % 50 == 0}, enable_tracking: {self.enable_tracking}")
    
    def log_detailed_losses(self, outputs):
        """Log detailed loss information for debugging"""
        logger.info(f"[Iter {self.iter+1}/{self.max_iter}] Detailed Losses:")
        logger.info(f"  Total Loss: {outputs.get('total_loss', 0):.4f}")
        
        # Standard YOLOX losses
        logger.info(f"  IoU Loss: {outputs.get('iou_loss', 0):.4f}")
        logger.info(f"  Obj Loss: {outputs.get('obj_loss', 0):.4f}")
        logger.info(f"  Cls Loss: {outputs.get('cls_loss', 0):.4f}")
        logger.info(f"  L1 Loss: {outputs.get('l1_loss', 0):.4f}")
        
        # UniTrack Unitrack loss
        if 'unitrack_loss' in outputs:
            logger.info(f"  Unitrack Loss: {outputs['unitrack_loss']:.4f}")
            logger.info(f"  Unitrack enabled: {getattr(self.model.head, 'enable_unitrack', False)}")
        
        # Tracking info
        if hasattr(self, 'tracker') and self.tracker is not None:
            logger.info(f"  Active Tracks: {len(self.tracker.tracked_stracks)}")
            logger.info(f"  Lost Tracks: {len(self.tracker.lost_stracks)}")
            logger.info(f"  Frame ID: {self.tracker.frame_id}")
    
    def visualize_tracking_results(self, inps, targets, outputs):
        """Visualize tracking results for debugging"""
        logger.info(f"DEBUG - Starting visualization for iter {self.iter}")
        try:
            import cv2
            import numpy as np
            
            # Process first image in batch
            img = inps[0].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            
            # Denormalize image (assuming standard ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img * std + mean) * 255
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Get model predictions for visualization
            predictions = self.get_model_predictions(inps[0:1], targets[0:1] if targets is not None else None)
            
            # Draw ground truth targets
            if targets is not None and len(targets) > 0:
                target = targets[0].cpu().numpy()
                valid_targets = 0
                for i in range(len(target)):
                    # Check if target is valid (not padding)
                    if target[i, 1] == 0 and target[i, 2] == 0 and target[i, 3] == 0 and target[i, 4] == 0:
                        continue
                    
                    valid_targets += 1
                    # Convert from center format to corner format
                    x, y, w, h = target[i, 1:5]
                    track_id = int(target[i, 5]) if len(target[i]) > 5 else 0
                    
                    # Scale to image size
                    img_h, img_w = img.shape[:2]
                    x = int(x * img_w / self.input_size[1])
                    y = int(y * img_h / self.input_size[0])
                    w = int(w * img_w / self.input_size[1])
                    h = int(h * img_h / self.input_size[0])
                    
                    # Draw bounding box (GREEN for ground truth)
                    cv2.rectangle(img, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
                    cv2.putText(img, f"GT:{track_id}", (x - w//2, y - h//2 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add text info
                cv2.putText(img, f"Iter: {self.iter+1}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, f"Valid Targets: {valid_targets}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add loss information
                if 'total_loss' in outputs:
                    cv2.putText(img, f"Total Loss: {outputs['total_loss']:.4f}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if 'unitrack_loss' in outputs:
                    cv2.putText(img, f"Unitrack Loss: {outputs['unitrack_loss']:.4f}", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw model predictions
            if predictions is not None and len(predictions) > 0:
                num_predictions = len(predictions)
                cv2.putText(img, f"Predictions: {num_predictions}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                for i, pred in enumerate(predictions):
                    if len(pred) >= 5:  # [x1, y1, x2, y2, score, class]
                        x1, y1, x2, y2, score = pred[:5]
                        
                        # Draw bounding box (RED for predictions)
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(img, f"P:{score:.2f}", (int(x1), int(y1) - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw tracker information if available
            if hasattr(self, 'tracker') and self.tracker is not None:
                active_tracks = len(self.tracker.tracked_stracks)
                lost_tracks = len(self.tracker.lost_stracks)
                cv2.putText(img, f"Active Tracks: {active_tracks}", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(img, f"Lost Tracks: {lost_tracks}", (10, 210), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw active tracks (BLUE for tracker results)
                for track in self.tracker.tracked_stracks:
                    if track.is_activated:
                        tlbr = track.tlbr
                        cv2.rectangle(img, (int(tlbr[0]), int(tlbr[1])), 
                                    (int(tlbr[2]), int(tlbr[3])), (255, 0, 0), 2)
                        cv2.putText(img, f"T:{track.track_id}", (int(tlbr[0]), int(tlbr[1]) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Save visualization
            save_path = f"debug_tracking_iter_{self.iter}.jpg"
            cv2.imwrite(save_path, img)
            logger.info(f"Saved tracking visualization to {save_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create tracking visualization: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
    
    def get_model_predictions(self, inps, targets):
        """
        Get model predictions for visualization
        
        Args:
            inps: Input images tensor [1, C, H, W]
            targets: Target tensor [1, max_labels, 6] (can be None)
            
        Returns:
            predictions: List of detection arrays [x1, y1, x2, y2, score, class]
        """
        try:
            # Store original training state
            was_training = self.model.training
            
            # Store original model dtype
            original_dtype = next(self.model.parameters()).dtype
            
            # Switch to eval mode for inference
            self.model.eval()
            
            with torch.no_grad():
                # Convert input to float32 for stable inference
                inps_inference = inps.float()
                
                # Temporarily convert model to float32 for inference
                self.model.float()
                
                # Run inference (no targets needed for inference)
                outputs = self.model(inps_inference)
                
                # Post-process outputs to get detections
                from yolox.utils import postprocess
                detections = postprocess(
                    outputs, 
                    num_classes=self.exp.num_classes, 
                    conf_thre=0.3,  # Lower threshold to see more detections
                    nms_thre=0.5
                )
                
                if detections[0] is not None:
                    # Convert to numpy and return
                    det = detections[0].cpu().numpy()
                    logger.info(f"DEBUG - Got {len(det)} predictions with max score: {det[:, 4].max():.3f}")
                    return det
                else:
                    logger.info("DEBUG - No predictions from model")
                    return []
                    
        except Exception as e:
            logger.warning(f"Error in prediction extraction: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
            return []
        finally:
            # Restore training mode and dtype
            if was_training:
                self.model.train()
            
            # Restore model to original dtype
            try:
                if original_dtype == torch.float16:
                    self.model.half()
                else:
                    self.model.float()
            except:
                # Fallback to training data type
                if self.data_type == torch.float16:
                    self.model.half()
                else:
                    self.model.float()

    def process_batch_with_tracking(self, inps, targets):
        """
        Process batch with tracking to assign real track IDs for Unitrack loss
        
        Args:
            inps: Input images tensor [B, C, H, W]
            targets: Target tensor [B, max_labels, 6] where last column is track_id
            
        Returns:
            targets: Updated targets with tracker-assigned IDs (maintains 6 columns)
        """
        if not self.enable_tracking:
            # If tracking is disabled, still keep original track_ids for Unitrack loss
            return targets
            
        # Initialize tracker if needed
        if not hasattr(self, 'tracker') or self.tracker is None:
            self.tracker = BYTETracker(self.tracking_args)
            
        # Debug: Log target information
        logger.info(f"DEBUG - Input targets shape: {targets.shape}")
        logger.info(f"DEBUG - Input targets non-zero count: {(targets.sum(dim=2) > 0).sum()}")
        if targets.shape[0] > 0:
            logger.info(f"DEBUG - First batch target sample: {targets[0][:3]}")
            
        batch_size = inps.shape[0]
        updated_targets = targets.clone()
        
        # Process each image in the batch
        for batch_idx in range(batch_size):
            img = inps[batch_idx]
            target = targets[batch_idx]
            
            # Skip if no valid targets in this batch
            valid_mask = (target[:, 4] > 0)  # Check if class > 0
            if not valid_mask.any():
                continue
            
            # Get detections from model in inference mode
            detections = self.get_detections_from_model(img)
            
            if detections is not None and len(detections) > 0:
                # Update tracker with detections
                self.frame_count += 1
                img_info = (self.input_size[0], self.input_size[1])
                
                # Run ByteTracker
                try:
                    track_results = self.tracker.update(detections, img_info, self.input_size)
                    
                    # Update targets with track IDs (maintain 6 columns)
                    updated_targets[batch_idx] = self.assign_track_ids_to_targets(
                        target, track_results, detections
                    )
                except Exception as e:
                    logger.warning(f"Tracker update failed: {e}")
                    # Keep original targets if tracking fails
                    pass
        
        return updated_targets

    def get_detections_from_model(self, img):
        """
        Get detections from model in inference mode
        
        Args:
            img: Input image tensor [C, H, W]
            
        Returns:
            detections: numpy array [N, 5] with [x1, y1, x2, y2, score]
        """
        # Store original training state and dtype
        was_training = self.model.training
        original_dtype = next(self.model.parameters()).dtype
        
        # Switch to eval mode
        self.model.eval()
        
        try:
            with torch.no_grad():
                # Add batch dimension and ensure correct dtype
                img_batch = img.unsqueeze(0)
                
                # Convert to float32 for stable inference if needed
                if self.data_type == torch.float16:
                    img_batch = img_batch.float()
                    # Temporarily convert model to float32 for inference
                    self.model.float()
                
                # Run inference (no targets needed)
                outputs = self.model(img_batch)
                
                # Post-process outputs
                detections = postprocess(
                    outputs, 
                    num_classes=self.exp.num_classes, 
                    conf_thre=0.3,  # Higher threshold for more reliable detections
                    nms_thre=0.5
                )
                
                if detections[0] is not None:
                    # Convert to numpy and format for ByteTracker
                    det = detections[0].cpu().numpy()
                    
                    # Format: [x1, y1, x2, y2, score] - remove class column for ByteTracker
                    return det[:, :5]
                else:
                    return np.empty((0, 5))
                    
        except Exception as e:
            logger.warning(f"Error in detection extraction: {e}")
            return np.empty((0, 5))
        finally:
            # Restore training mode
            if was_training:
                self.model.train()
            
            # Restore original dtype
            if original_dtype != next(self.model.parameters()).dtype:
                if original_dtype == torch.float16:
                    self.model.half()
                else:
                    self.model.float()

    def assign_track_ids_to_targets(self, target, track_results, detections):
        """
        Assign track IDs from ByteTracker to training targets
        
        Args:
            target: Original target tensor [max_labels, 6]
            track_results: List of STrack objects from ByteTracker
            detections: Detection array used for tracking
            
        Returns:
            updated_target: Target tensor with updated track IDs (maintains 6 columns)
        """
        updated_target = target.clone()
        
        if not track_results or len(track_results) == 0:
            return updated_target
        
        # Find valid targets (non-zero classes)
        valid_mask = (target[:, 4] > 0)  # Class > 0
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return updated_target
        
        # Simple assignment strategy: assign track IDs based on detection order
        # This is a simplified approach - in practice, you'd want proper association
        num_assignments = min(len(valid_indices), len(track_results))
        
        for i in range(num_assignments):
            target_idx = valid_indices[i]
            track_id = track_results[i].track_id
            
            # Assign track ID to the target (column 5)
            updated_target[target_idx, 5] = float(track_id)
        
        # For remaining targets without track assignments, use original track IDs
        # This ensures we don't lose track information
        for i in range(num_assignments, len(valid_indices)):
            target_idx = valid_indices[i]
            # Keep original track ID if no new assignment
            if target[target_idx, 5] == 0:
                # Assign a unique fallback track ID
                updated_target[target_idx, 5] = float(1000 + i)  # Fallback ID
        
        return updated_target

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
            # occupancy_for_ckpt(model)  # Function not available, skip
            pass

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
            # all_reduce_norm(self.model)  # Function might not be available
            self.evaluate_and_save_model()

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        # Temporarily switch to eval mode
        evalmodel.eval()
        
        ap50_95, ap50, summary = self.evaluator.evaluate(
            evalmodel, self.is_distributed, self.args.fp16
        )
        
        # Switch back to train mode
        self.model.train()

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)
        
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        # Save latest checkpoint
        self.save_ckpt("last_epoch")
        
        # Save best checkpoint if improved
        if update_best_ckpt:
            self.save_ckpt("best_ckpt", update_best_ckpt=True)

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth.tar")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            
            # Note: lr_scheduler is not saved/loaded as it's stateless and reconstructed from config
            
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
                ckpt = torch.load(ckpt_file, map_location=self.device)
                if isinstance(ckpt, dict) and "model" in ckpt:
                    model = load_ckpt(model, ckpt["model"])
                else:
                    model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model


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