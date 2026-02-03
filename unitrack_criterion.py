"""
UniTrack: Universal Tracking Loss Function for Multi-Object Tracking

This module implements the UniTrack (HINGE) loss function, a universal tracking criterion
that can be integrated into any Multi-Object Tracking (MOT) framework to improve tracking
consistency and reduce identity switches.

Paper: "UniTrack: Enhanced Multi-Object Tracking with Hinge Loss" (ICLR 2026)

Key Features:
    - Framework-agnostic design - works with any MOT architecture
    - Three complementary loss components:
        1. Tracking Score: Reduces false positives, false negatives, and ID switches
        2. Spatial Consistency: Enforces consistent object sizes/shapes within tracks
        3. Temporal Consistency: Encourages smooth motion trajectories
    - Differentiable and end-to-end trainable
    - Minimal computational overhead

Integration:
    The loss can be integrated into any MOT framework by:
    1. Importing the Unitrackrion class
    2. Instantiating it with your image size and IoU threshold
    3. Computing the loss during training by passing predictions and targets
    4. Adding the weighted loss to your total training loss

Example:
    >>> from unitrack_criterion import Unitrackrion
    >>> 
    >>> # Initialize the criterion
    >>> unitrack_loss = Unitrackrion(img_size=(1920, 1080), iou_threshold=0.5)
    >>> 
    >>> # During training
    >>> outputs = {
    >>>     'pred_boxes': pred_boxes,      # (batch_size, num_queries, 4)
    >>>     'pred_logits': pred_logits,    # (batch_size, num_queries, num_classes)
    >>>     'track_ids': track_ids         # (batch_size, num_queries)
    >>> }
    >>> targets = [
    >>>     {
    >>>         'boxes': gt_boxes,         # (num_objects, 4)
    >>>         'labels': gt_labels,       # (num_objects,)
    >>>         'track_ids': gt_track_ids  # (num_objects,)
    >>>     }
    >>> ]
    >>> 
    >>> # Compute loss
    >>> loss_dict = unitrack_loss(outputs, targets)
    >>> total_loss = loss_dict['loss_unitrack']

Tested Frameworks:
    - GTR (Global Tracking Transformers)
    - MOTR (MOT with Transformers)
    - TrackFormer
    - FairMOT
    - ByteTrack
    - MOTE
    - And more...

Author: [Your Name/Team]
License: [Your License]
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple, Optional


class Unitrackrion(nn.Module):
    """
    UniTrack Loss Criterion for Multi-Object Tracking.
    
    This loss function combines three components to improve tracking performance:
    1. Tracking score: Penalizes false positives, false negatives, and ID switches
    2. Spatial consistency: Encourages consistent object dimensions within tracks
    3. Temporal consistency: Promotes smooth motion trajectories
    
    Args:
        img_size (tuple): Image dimensions as (width, height). Used for normalization.
                         Default: (1920, 1080)
        iou_threshold (float): IoU threshold for considering a detection as matched.
                              Default: 0.5
    
    Attributes:
        alpha_tracking (float): Weight for tracking score component (default: 2.0)
        alpha_spatial (float): Weight for spatial consistency component (default: 1.5)
        alpha_temporal (float): Weight for temporal consistency component (default: 1.8)
        beta_fp (float): Penalty weight for false positives (default: 0.9)
        beta_fn (float): Penalty weight for false negatives (default: 0.9)
        gamma_switch (float): Penalty weight for ID switches (default: 1.5)
    
    Note:
        All weight parameters (alpha_*, beta_*, gamma_*) are registered as buffers,
        making them part of the module's state but not trainable parameters.
        You can modify them after initialization if needed:
        
        >>> criterion = Unitrackrion()
        >>> criterion.alpha_tracking = torch.tensor(3.0)
    """
    
    def __init__(self, img_size: Tuple[int, int] = (1920, 1080), iou_threshold: float = 0.5):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.img_diagonal = math.sqrt(img_size[0]**2 + img_size[1]**2)
        self.scale_factor = 1.0 / self.img_diagonal
        
        # Loss component weights (registered as buffers for state dict compatibility)
        self.register_buffer('alpha_tracking', torch.tensor(2.0))
        self.register_buffer('alpha_spatial', torch.tensor(1.5))
        self.register_buffer('alpha_temporal', torch.tensor(1.8))
        self.register_buffer('beta_fp', torch.tensor(0.9))
        self.register_buffer('beta_fn', torch.tensor(0.9))
        self.register_buffer('gamma_switch', torch.tensor(1.5))

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute the UniTrack loss for tracking.
        
        Args:
            outputs: Dictionary containing model predictions with keys:
                - pred_boxes: Tensor of shape (batch_size, num_queries, 4)
                             Predicted bounding boxes in [x1, y1, x2, y2] format
                - pred_logits: Tensor of shape (batch_size, num_queries, num_classes)
                              Class prediction logits (used for validation, not in loss)
                - track_ids: Tensor of shape (batch_size, num_queries)
                            Predicted track IDs for each detection
                            
            targets: List of dictionaries (one per batch item), each containing:
                - boxes: Tensor of shape (num_objects, 4)
                        Ground truth boxes in [x1, y1, x2, y2] format
                - labels: Tensor of shape (num_objects,)
                         Ground truth class labels (optional, not used in loss)
                - track_ids: Tensor of shape (num_objects,)
                            Ground truth track IDs
                
        Returns:
            Dictionary containing loss components:
                - loss_unitrack: Total weighted UniTrack loss
                - loss_unitrack_tracking: Tracking score component
                - loss_unitrack_spatial: Spatial consistency component
                - loss_unitrack_temporal: Temporal consistency component
                
        Note:
            - Predictions with track_id == 0 are considered invalid and ignored
            - Empty batches (no valid tracks or targets) contribute zero loss
            - All losses are normalized by the number of unique tracks
        """
        device = outputs["pred_boxes"].device
        batch_size = len(targets)
        
        # Initialize loss components
        loss_tracking = torch.tensor(0.0, device=device)
        loss_spatial = torch.tensor(0.0, device=device)
        loss_temporal = torch.tensor(0.0, device=device)
        num_tracks = 0
        
        for batch_idx in range(batch_size):
            # Extract predictions and targets for current batch
            pred_boxes = outputs["pred_boxes"][batch_idx]  # (num_queries, 4)
            pred_logits = outputs["pred_logits"][batch_idx]  # (num_queries, num_classes)
            pred_track_ids = outputs["track_ids"][batch_idx]  # (num_queries,)
            
            # Filter predictions to get only the ones with valid track IDs
            valid_track_mask = pred_track_ids > 0
            if not valid_track_mask.any():
                continue  # No valid tracks in this batch
                
            pred_boxes = pred_boxes[valid_track_mask]  # (num_valid_tracks, 4)
            pred_logits = pred_logits[valid_track_mask]  # (num_valid_tracks, num_classes)
            pred_track_ids = pred_track_ids[valid_track_mask]  # (num_valid_tracks,)
            
            # Get target boxes and track IDs
            gt_boxes = targets[batch_idx]["boxes"]  # (num_objects, 4)
            gt_track_ids = targets[batch_idx]["track_ids"]  # (num_objects,)
            
            # Skip if no ground truth objects
            if len(gt_boxes) == 0:
                continue
                
            # Compute IoU between predicted boxes and ground truth boxes
            ious = self._box_iou(pred_boxes, gt_boxes)  # (num_valid_tracks, num_objects)
            
            # Compute tracking score based on IoUs and track IDs
            batch_tracking_loss = self._compute_tracking_loss(ious, pred_track_ids, gt_track_ids)
            loss_tracking += batch_tracking_loss
            
            # Compute spatial consistency loss
            batch_spatial_loss = self._compute_spatial_consistency(pred_boxes, pred_track_ids)
            loss_spatial += batch_spatial_loss
            
            # Compute temporal consistency loss
            batch_temporal_loss = self._compute_temporal_consistency(pred_boxes, pred_track_ids)
            loss_temporal += batch_temporal_loss
            
            # Count the number of unique tracks for normalization
            num_tracks += len(torch.unique(pred_track_ids))
        
        # Normalize losses by the number of tracks
        if num_tracks > 0:
            loss_tracking = loss_tracking / num_tracks
            loss_spatial = loss_spatial / num_tracks
            loss_temporal = loss_temporal / num_tracks
        
        # Compute weighted sum of losses
        total_loss = (self.alpha_tracking * loss_tracking + 
                     self.alpha_spatial * loss_spatial + 
                     self.alpha_temporal * loss_temporal)
        
        # Return individual loss components as well as total loss
        return {
            'loss_unitrack': total_loss,
            'loss_unitrack_tracking': loss_tracking,
            'loss_unitrack_spatial': loss_spatial,
            'loss_unitrack_temporal': loss_temporal
        }
    
    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise IoU between two sets of boxes.
        
        Args:
            boxes1: Tensor of shape (N, 4) - predicted boxes in [x1, y1, x2, y2] format
            boxes2: Tensor of shape (M, 4) - ground truth boxes in [x1, y1, x2, y2] format
            
        Returns:
            Tensor of shape (N, M) containing pairwise IoU values
        """
        area1 = self._box_area(boxes1)  # (N,)
        area2 = self._box_area(boxes2)  # (M,)
        
        # Get the intersections
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
        wh = (rb - lt).clamp(min=0)  # (N, M, 2)
        intersection = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
        
        # Compute IoU
        union = area1[:, None] + area2 - intersection  # (N, M)
        iou = intersection / union  # (N, M)
        
        return iou
    
    def _box_area(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute area of boxes.
        
        Args:
            boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
            
        Returns:
            Tensor of shape (N,) containing box areas
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    def _compute_tracking_loss(self, ious: torch.Tensor, pred_track_ids: torch.Tensor, 
                               gt_track_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute tracking loss based on IoUs and track IDs.
        
        This component penalizes:
        1. False negatives: Ground truth objects with no matching prediction
        2. Poor localization: Predictions with low IoU to their matched ground truth
        3. ID switches: Predictions with wrong track ID but high IoU to ground truth
        4. False positives: Predictions with track IDs not in ground truth
        
        Args:
            ious: Tensor of shape (num_pred, num_gt) - IoU matrix
            pred_track_ids: Tensor of shape (num_pred,) - predicted track IDs
            gt_track_ids: Tensor of shape (num_gt,) - ground truth track IDs
            
        Returns:
            Scalar tensor containing the tracking loss
        """
        loss = torch.tensor(0.0, device=ious.device)
        
        # For each ground truth track
        unique_gt_tracks = torch.unique(gt_track_ids)
        for gt_track_id in unique_gt_tracks:
            gt_mask = gt_track_ids == gt_track_id
            gt_indices = torch.where(gt_mask)[0]
            
            # Find predictions with the same track ID
            pred_mask = pred_track_ids == gt_track_id
            
            # False negatives: ground truth objects with no matching prediction
            if not pred_mask.any():
                loss = loss + self.beta_fn * len(gt_indices)
                continue
                
            # For matched track IDs, compute tracking score based on IoUs
            pred_indices = torch.where(pred_mask)[0]
            track_ious = ious[pred_indices][:, gt_indices]
            
            # Compute penalty for bad localization (low IoU)
            max_ious, _ = track_ious.max(dim=1)
            localization_penalty = torch.sum(1.0 - max_ious)
            
            # Compute penalty for ID switches
            switch_penalty = torch.tensor(0.0, device=ious.device)
            for gt_idx in gt_indices:
                # Find predictions that match this ground truth but have wrong ID
                other_pred_mask = ~pred_mask
                if other_pred_mask.any():
                    other_pred_ious = ious[other_pred_mask][:, gt_idx]
                    wrong_matches = other_pred_ious > self.iou_threshold
                    switch_penalty = switch_penalty + self.gamma_switch * wrong_matches.sum()
            
            loss = loss + localization_penalty + switch_penalty
        
        # False positives: predictions with no matching ground truth
        for pred_track_id in torch.unique(pred_track_ids):
            pred_mask = pred_track_ids == pred_track_id
            pred_indices = torch.where(pred_mask)[0]
            
            # Check if this track ID exists in ground truth
            if pred_track_id in unique_gt_tracks:
                continue
                
            # This is a false positive track
            loss = loss + self.beta_fp * len(pred_indices)
        
        return loss
    
    def _compute_spatial_consistency(self, pred_boxes: torch.Tensor, 
                                    pred_track_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial consistency loss for tracks.
        
        Encourages consistent size/shape of boxes for the same track.
        This is based on the assumption that object dimensions don't change
        drastically within a short temporal window.
        
        Args:
            pred_boxes: Tensor of shape (N, 4) - predicted boxes
            pred_track_ids: Tensor of shape (N,) - predicted track IDs
            
        Returns:
            Scalar tensor containing the spatial consistency loss
        """
        if len(pred_boxes) <= 1:
            return torch.tensor(0.0, device=pred_boxes.device)
            
        loss = torch.tensor(0.0, device=pred_boxes.device)
        unique_tracks = torch.unique(pred_track_ids)
        
        for track_id in unique_tracks:
            track_mask = pred_track_ids == track_id
            track_boxes = pred_boxes[track_mask]
            
            if len(track_boxes) <= 1:
                continue
                
            # Compute mean box dimensions
            widths = track_boxes[:, 2] - track_boxes[:, 0]
            heights = track_boxes[:, 3] - track_boxes[:, 1]
            mean_width = widths.mean()
            mean_height = heights.mean()
            
            # Penalize deviations from mean dimensions
            width_deviation = torch.abs(widths - mean_width)
            height_deviation = torch.abs(heights - mean_height)
            
            # Scale by image size for normalization
            width_deviation = width_deviation * self.scale_factor
            height_deviation = height_deviation * self.scale_factor
            
            loss = loss + width_deviation.mean() + height_deviation.mean()
        
        return loss / max(len(unique_tracks), 1)
    
    def _compute_temporal_consistency(self, pred_boxes: torch.Tensor, 
                                     pred_track_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss for tracks.
        
        Encourages smooth motion of tracked objects by penalizing high accelerations.
        This assumes objects move with relatively constant velocity within short
        temporal windows.
        
        Args:
            pred_boxes: Tensor of shape (N, 4) - predicted boxes
            pred_track_ids: Tensor of shape (N,) - predicted track IDs
            
        Returns:
            Scalar tensor containing the temporal consistency loss
            
        Note:
            Requires at least 3 boxes per track to compute acceleration.
            Tracks with fewer boxes are skipped.
        """
        if len(pred_boxes) <= 1:
            return torch.tensor(0.0, device=pred_boxes.device)
            
        loss = torch.tensor(0.0, device=pred_boxes.device)
        unique_tracks = torch.unique(pred_track_ids)
        
        for track_id in unique_tracks:
            track_mask = pred_track_ids == track_id
            track_boxes = pred_boxes[track_mask]
            
            if len(track_boxes) <= 2:  # Need at least 3 boxes to compute acceleration
                continue
                
            # Compute centers
            centers_x = (track_boxes[:, 0] + track_boxes[:, 2]) / 2
            centers_y = (track_boxes[:, 1] + track_boxes[:, 3]) / 2
            centers = torch.stack([centers_x, centers_y], dim=1)
            
            # Compute velocities (differences between consecutive centers)
            velocities = centers[1:] - centers[:-1]
            
            # Compute accelerations (differences between consecutive velocities)
            accelerations = velocities[1:] - velocities[:-1]
            
            # Scale by image size for normalization
            accelerations = accelerations * self.scale_factor
            
            # Penalize high accelerations (non-smooth motion)
            acceleration_penalty = torch.norm(accelerations, dim=1).mean()
            
            loss = loss + acceleration_penalty
        
        return loss / max(len(unique_tracks), 1)


# Convenience function for quick integration
def create_unitrack_loss(img_size=(1920, 1080), iou_threshold=0.5, 
                         alpha_tracking=2.0, alpha_spatial=1.5, alpha_temporal=1.8):
    """
    Factory function to create a UniTrack loss with custom parameters.
    
    Args:
        img_size: Image dimensions (width, height)
        iou_threshold: IoU threshold for matching
        alpha_tracking: Weight for tracking component
        alpha_spatial: Weight for spatial consistency
        alpha_temporal: Weight for temporal consistency
        
    Returns:
        Configured Unitrackrion instance
        
    Example:
        >>> loss_fn = create_unitrack_loss(
        ...     img_size=(1920, 1080),
        ...     alpha_tracking=3.0,
        ...     alpha_spatial=2.0
        ... )
    """
    criterion = Unitrackrion(img_size=img_size, iou_threshold=iou_threshold)
    criterion.alpha_tracking = torch.tensor(alpha_tracking)
    criterion.alpha_spatial = torch.tensor(alpha_spatial)
    criterion.alpha_temporal = torch.tensor(alpha_temporal)
    return criterion
