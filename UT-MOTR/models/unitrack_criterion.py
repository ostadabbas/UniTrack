import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import math

class Unitrackrion(nn.Module):
    """PyTorch implementation of Unitrack metric for MOTR tracking."""
    
    def __init__(self, img_size=(1920, 1080), iou_threshold=0.5):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.img_diagonal = math.sqrt(img_size[0]**2 + img_size[1]**2)
        self.scale_factor = 1.0 / self.img_diagonal
        
        # Dynamic weighting parameters (can be made learnable if needed)
        self.register_buffer('alpha_tracking', torch.tensor(2.0))
        self.register_buffer('alpha_spatial', torch.tensor(1.5))
        self.register_buffer('alpha_temporal', torch.tensor(1.8))
        self.register_buffer('beta_fp', torch.tensor(0.9))
        self.register_buffer('beta_fn', torch.tensor(0.9))
        self.register_buffer('gamma_switch', torch.tensor(1.5))

    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU between two sets of boxes.
        MOTR uses (cx, cy, w, h) format, so convert to (x1, y1, x2, y2) first.
        """
        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
        def cxcywh_to_xyxy(boxes):
            cx, cy, w, h = boxes.unbind(-1)
            return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)
        
        boxes1_xyxy = cxcywh_to_xyxy(boxes1)
        boxes2_xyxy = cxcywh_to_xyxy(boxes2)
        
        # Compute intersection
        x1_max = torch.max(boxes1_xyxy[:, None, 0], boxes2_xyxy[None, :, 0])
        y1_max = torch.max(boxes1_xyxy[:, None, 1], boxes2_xyxy[None, :, 1])
        x2_min = torch.min(boxes1_xyxy[:, None, 2], boxes2_xyxy[None, :, 2])
        y2_min = torch.min(boxes1_xyxy[:, None, 3], boxes2_xyxy[None, :, 3])
        
        inter = torch.clamp(x2_min - x1_max, min=0) * torch.clamp(y2_min - y1_max, min=0)
        
        # Compute areas
        area1 = boxes1[:, 2] * boxes1[:, 3]  # w * h for (cx, cy, w, h)
        area2 = boxes2[:, 2] * boxes2[:, 3]
        
        # Compute union
        union = (area1[:, None] + area2[None, :] - inter + 1e-6)
        
        return inter / union

    def _compute_spatial_consistency(self, boxes: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Compute spatial consistency within a frame using your original approach."""
        if boxes.size(0) < 2:
            return torch.tensor(0.0, device=boxes.device)
        
        # Compute pairwise distances between centers (your original method)
        diffs = centers.unsqueeze(1) - centers.unsqueeze(0)
        distances = torch.norm(diffs, dim=2) * self.scale_factor
        
        # Simple spatial consistency: average normalized distance (preserve original theory)
        # Remove diagonal (self-distances)
        mask = ~torch.eye(len(centers), dtype=torch.bool, device=centers.device)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=boxes.device)
        
        avg_distance = distances[mask].mean()
        # Convert to error (higher distance = higher error)
        spatial_error = torch.clamp(avg_distance, 0.0, 1.0)
        
        return spatial_error

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute Unitrack loss from MOTR outputs and targets.
        
        Args:
            outputs: Dict containing:
                - 'pred_boxes': [batch_size, num_queries, 4] predicted boxes (cx,cy,w,h)
                - 'pred_logits': [batch_size, num_queries, num_classes] predicted classes  
                - 'track_ids': [batch_size, num_queries] predicted track IDs
            targets: List of dicts containing:
                - 'boxes': [num_target_boxes, 4] ground truth boxes
                - 'labels': [num_target_boxes] ground truth labels
                - 'track_ids': [num_target_boxes] ground truth track IDs
                
        Returns:
            Dict containing various components of the Unitrack loss
        """
        device = outputs['pred_boxes'].device
        losses = {}
        
        # Process each batch independently
        for batch_idx in range(outputs['pred_boxes'].shape[0]):
            pred_boxes = outputs['pred_boxes'][batch_idx]  # [num_queries, 4] in (cx, cy, w, h)
            pred_logits = outputs['pred_logits'][batch_idx]  # [num_queries, num_classes]
            pred_ids = outputs['track_ids'][batch_idx]  # [num_queries]
            
            # Get target values for this batch
            gt_boxes = targets[batch_idx]['boxes']  # [num_targets, 4] in (cx, cy, w, h)
            gt_ids = targets[batch_idx]['track_ids']  # [num_targets]
            
            # Compute IoU matrix
            ious = self._compute_iou(pred_boxes, gt_boxes)
            
            # Compute centers for spatial consistency (cx, cy from MOTR format)
            centers = pred_boxes[:, :2]  # Already (cx, cy) in MOTR format
            
            # Compute spatial consistency
            spatial_error = self._compute_spatial_consistency(pred_boxes, centers)
            
            # Compute precision and recall based on IoU matching (preserve original theory)
            matches = (ious > self.iou_threshold).float()
            tp = matches.sum()
            fp = (matches.sum(dim=1) == 0).float().sum()
            fn = (matches.sum(dim=0) == 0).float().sum()
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
            
            # Simplified loss computation to avoid saturation (preserve core method)
            tracking_loss = 1.0 - f1_score  # Direct F1-based loss
            spatial_loss = spatial_error     # Direct spatial error
            
            # Skip temporal for now to remove global state contamination
            # This preserves training stability while keeping your core Unitrack theory
            temporal_loss = torch.tensor(0.0, device=device)
            
            # Weighted combination using your original weights
            total_loss = (
                self.alpha_tracking * tracking_loss +
                self.alpha_spatial * spatial_loss +
                self.alpha_temporal * temporal_loss
            ) / (self.alpha_tracking + self.alpha_spatial + self.alpha_temporal)
            
            losses[f'batch_{batch_idx}_unitrack'] = total_loss
        
        # Average losses across batch
        losses['loss_unitrack'] = torch.stack([v for k, v in losses.items() if 'batch_' in k]).mean()
        
        print("[DEBUG] Unitrack losses:", {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()})
        
        return losses
