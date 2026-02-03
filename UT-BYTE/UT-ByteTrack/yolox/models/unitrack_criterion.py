import torch
import torch.nn as nn
import math
from typing import Dict, List


class Unitrackrion(nn.Module):
    def __init__(self, img_size=(1920, 1080), iou_threshold=0.5):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.img_diagonal = math.sqrt(img_size[0]**2 + img_size[1]**2)
        self.scale_factor = 1.0 / self.img_diagonal
        self.register_buffer('alpha_tracking', torch.tensor(2.0))
        self.register_buffer('alpha_spatial', torch.tensor(1.5))
        self.register_buffer('alpha_temporal', torch.tensor(1.8))
        self.register_buffer('beta_fp', torch.tensor(0.9))
        self.register_buffer('beta_fn', torch.tensor(0.9))
        self.register_buffer('gamma_switch', torch.tensor(1.5))

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """Compute the Unitrack loss for tracking.
        
        Args:
            outputs: Dict containing:
                - pred_boxes: Tensor of shape (batch_size, num_queries, 4) with predicted boxes
                - pred_logits: Tensor of shape (batch_size, num_queries, num_classes) with class logits
                - track_ids: Tensor of shape (batch_size, num_queries) with predicted track IDs
            targets: List of dictionaries, each containing:
                - boxes: Tensor of shape (num_objects, 4) with ground truth boxes
                - labels: Tensor of shape (num_objects,) with ground truth class labels
                - track_ids: Tensor of shape (num_objects,) with ground truth track IDs
                
        Returns:
            Dict containing the Unitrack loss components
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
        
        # Check for NaN values and replace with zeros
        if torch.isnan(loss_tracking) or torch.isinf(loss_tracking):
            loss_tracking = torch.tensor(0.0, device=device)
        if torch.isnan(loss_spatial) or torch.isinf(loss_spatial):
            loss_spatial = torch.tensor(0.0, device=device)
        if torch.isnan(loss_temporal) or torch.isinf(loss_temporal):
            loss_temporal = torch.tensor(0.0, device=device)
        
        # Compute weighted sum of losses
        total_loss = (self.alpha_tracking * loss_tracking + 
                     self.alpha_spatial * loss_spatial + 
                     self.alpha_temporal * loss_temporal)
        
        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=device)
        
        # Return individual loss components as well as total loss
        return {
            'loss_unitrack': total_loss,
            'loss_unitrack_tracking': loss_tracking,
            'loss_unitrack_spatial': loss_spatial,
            'loss_unitrack_temporal': loss_temporal
        }
    
    def _box_iou(self, boxes1, boxes2):
        """
        Compute IoU between two sets of boxes.
        boxes1: (N, 4) - predicted boxes
        boxes2: (M, 4) - ground truth boxes
        Returns: (N, M) IoU matrix
        """
        area1 = self._box_area(boxes1)  # (N,)
        area2 = self._box_area(boxes2)  # (M,)
        
        # Get the intersections
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
        wh = (rb - lt).clamp(min=0)  # (N, M, 2)
        intersection = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
        
        # Compute IoU with numerical stability
        union = area1[:, None] + area2 - intersection  # (N, M)
        # Add small epsilon to avoid division by zero
        eps = 1e-6
        iou = intersection / (union + eps)  # (N, M)
        
        # Clamp to [0, 1] to handle numerical issues
        iou = torch.clamp(iou, 0.0, 1.0)
        
        return iou
    
    def _box_area(self, boxes):
        """
        Compute area of boxes.
        boxes: (N, 4) - [x1, y1, x2, y2] format
        Returns: (N,) areas
        """
        # Ensure boxes have positive width and height
        width = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0.0)
        height = torch.clamp(boxes[:, 3] - boxes[:, 1], min=0.0)
        return width * height
    
    def _compute_tracking_loss(self, ious, pred_track_ids, gt_track_ids):
        """
        Compute tracking loss based on IoUs and track IDs.
        ious: (num_pred, num_gt) IoU matrix
        pred_track_ids: (num_pred,) predicted track IDs
        gt_track_ids: (num_gt,) ground truth track IDs
        Returns: tracking loss
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
    
    def _compute_spatial_consistency(self, pred_boxes, pred_track_ids):
        """
        Compute spatial consistency loss for tracks.
        Encourages consistent size/shape of boxes for the same track.
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
    
    def _compute_temporal_consistency(self, pred_boxes, pred_track_ids):
        """
        Compute temporal consistency loss for tracks.
        Encourages smooth motion of tracked objects.
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