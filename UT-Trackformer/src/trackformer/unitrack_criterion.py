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
        self.register_buffer('alpha_tracking', torch.tensor(2.0)) # 2.0
        self.register_buffer('alpha_spatial', torch.tensor(1.5)) # 1.5
        self.register_buffer('alpha_temporal', torch.tensor(1.8)) # 1.8
        self.register_buffer('beta_fp', torch.tensor(0.9))
        self.register_buffer('beta_fn', torch.tensor(0.9))
        self.register_buffer('gamma_switch', torch.tensor(1.5))

    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU between two sets of boxes."""
        # boxes are in (x1, y1, w, h) format
        x1_1, y1_1 = boxes1[..., 0], boxes1[..., 1]
        x2_1, y2_1 = x1_1 + boxes1[..., 2], y1_1 + boxes1[..., 3]
        x1_2, y1_2 = boxes2[..., 0], boxes2[..., 1]
        x2_2 = x1_2 + boxes2[..., 2]
        y2_2 = y1_2 + boxes2[..., 3]
        
        # Compute intersection
        x1_max = torch.max(x1_1.unsqueeze(-1), x1_2.unsqueeze(0))
        y1_max = torch.max(y1_1.unsqueeze(-1), y1_2.unsqueeze(0))
        x2_min = torch.min(x2_1.unsqueeze(-1), x2_2.unsqueeze(0))
        y2_min = torch.min(y2_1.unsqueeze(-1), y2_2.unsqueeze(0))
        
        inter = torch.clamp(x2_min - x1_max, min=0) * torch.clamp(y2_min - y1_max, min=0)
        
        # Compute areas
        area1 = boxes1[..., 2] * boxes1[..., 3]
        area2 = boxes2[..., 2] * boxes2[..., 3]
        
        # Compute union
        union = (area1.unsqueeze(-1) + area2.unsqueeze(0) - inter + 1e-6)
        
        return inter / union

    def _compute_spatial_consistency(self, boxes: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Compute spatial consistency within a frame."""
        if boxes.size(0) < 2:
            return torch.tensor(0.0, device=boxes.device)
        
        # Compute pairwise distances between centers
        diffs = centers.unsqueeze(1) - centers.unsqueeze(0)
        distances = torch.norm(diffs, dim=2) * self.scale_factor
        
        # Create adjacency matrix (1 if objects are "close")
        adj_matrix = (distances < 0.1).float()
        adj_matrix.fill_diagonal_(0)
        
        # Compute clustering coefficient
        neighbor_counts = adj_matrix.sum(dim=1)
        valid_nodes = neighbor_counts >= 2
        
        if not valid_nodes.any():
            return torch.tensor(0.0, device=boxes.device)
        
        clustering_coefs = []
        for i in torch.where(valid_nodes)[0]:
            neighbors = adj_matrix[i].bool()
            neighbor_subgraph = adj_matrix[neighbors][:, neighbors]
            possible_connections = neighbor_counts[i] * (neighbor_counts[i] - 1) / 2
            actual_connections = neighbor_subgraph.sum() / 2
            if possible_connections > 0:
                clustering_coefs.append(actual_connections / possible_connections)
        
        return torch.stack(clustering_coefs).mean() if clustering_coefs else torch.tensor(0.0, device=boxes.device)

    def _compute_temporal_consistency(self, prev_boxes: torch.Tensor, curr_boxes: torch.Tensor,
                                    prev_ids: torch.Tensor, curr_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute temporal consistency between consecutive frames."""
        # If no previous boxes, return perfect consistency
        if len(prev_boxes) == 0 or len(curr_boxes) == 0:
            return {
                'temporal_score': torch.ones(1, device=curr_boxes.device),
                'id_switches': torch.zeros(1, device=curr_boxes.device)
            }
        
        # Compute IoU between previous and current boxes
        ious = self._compute_iou(prev_boxes, curr_boxes)
        
        # Find matching boxes based on IoU threshold
        matches = (ious > self.iou_threshold)
        
        # Count ID switches
        id_switches = 0
        for i in range(len(prev_boxes)):
            for j in range(len(curr_boxes)):
                if matches[i, j] and prev_ids[i] != curr_ids[j]:
                    id_switches += 1
        
        # Compute temporal consistency score
        temporal_score = 1.0 - (id_switches * self.gamma_switch) / (len(prev_boxes) + 1e-6)
        temporal_score = torch.clamp(temporal_score, min=0.0)
        
        return {
            'temporal_score': temporal_score.to(prev_boxes.device),
            'id_switches': torch.tensor(id_switches, device=prev_boxes.device, dtype=torch.float32)
        }

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute Unitrack loss from MOTR outputs and targets.
        
        Args:
            outputs: Dict containing:
                - 'pred_boxes': [batch_size, num_queries, 4] predicted boxes (x1,y1,w,h)
                - 'pred_logits': [batch_size, num_queries, num_classes] predicted classes
                - 'track_ids': [batch_size, max_ids] ground truth track IDs
                - 'track_boxes': [batch_size, max_ids, 4] predicted boxes for track queries
                - 'track_logits': [batch_size, max_ids, num_classes] predicted logits for track queries
            targets: List of dicts containing:
                - 'boxes': [num_target_boxes, 4] ground truth boxes
                - 'labels': [num_target_boxes] ground truth labels
                - 'track_ids': [num_target_boxes] ground truth track IDs
                
        Returns:
            Dict containing various components of the Unitrack loss
        """
        device = outputs['pred_boxes'].device
        losses = {}
        
        # Skip if required outputs are missing
        if not all(k in outputs for k in ['track_ids', 'track_boxes', 'track_logits']):
            return {'loss_unitrack': torch.tensor(0.0, device=device)}
        
        # Process each batch independently
        for batch_idx in range(outputs['track_ids'].shape[0]):
            # Get track-specific tensors
            gt_ids = outputs['track_ids'][batch_idx]  # [max_ids]
            pred_boxes = outputs['track_boxes'][batch_idx]  # [max_ids, 4]
            pred_logits = outputs['track_logits'][batch_idx]  # [max_ids, num_classes]
            
            # Remove padding (-1s) from track IDs and corresponding tensors
            valid_mask = gt_ids != -1
            gt_ids = gt_ids[valid_mask]
            pred_boxes = pred_boxes[valid_mask]
            pred_logits = pred_logits[valid_mask]
            
            # Skip if no valid predictions
            if len(pred_boxes) == 0:
                continue
                
            # Get target boxes for IoU computation
            gt_boxes = targets[batch_idx]['boxes']  # [num_targets, 4]
            
            # Compute IoU matrix between predicted and target boxes
            ious = self._compute_iou(pred_boxes, gt_boxes)  # [num_pred, num_targets]
            
            # Compute precision and recall based on IoU matching
            matches = (ious > self.iou_threshold).float()
            tp = matches.sum()
            fp = (matches.sum(dim=1) == 0).float().sum()
            fn = (matches.sum(dim=0) == 0).float().sum()
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
            
            # Compute tracking score with penalties
            exponent = -self.beta_fp * fp / (pred_boxes.size(0) + 1e-6) \
                      -self.beta_fn * fn / (gt_boxes.size(0) + 1e-6)
            tracking_score = f1_score * torch.exp(torch.clamp(exponent, min=-20))
            
            # Compute spatial consistency
            centers = torch.stack([
                pred_boxes[:, 0] + pred_boxes[:, 2]/2,
                pred_boxes[:, 1] + pred_boxes[:, 3]/2
            ], dim=1)
            
            spatial_error = self._compute_spatial_consistency(pred_boxes, centers)
            spatial_score = 1 - spatial_error
            
            # Compute temporal consistency
            if hasattr(self, 'prev_boxes') and hasattr(self, 'prev_ids'):
                temp_metrics = self._compute_temporal_consistency(
                    self.prev_boxes, pred_boxes,
                    self.prev_ids, gt_ids
                )
                temporal_score = temp_metrics['temporal_score']
            else:
                temporal_score = torch.tensor(1.0, device=device)
            
            # Save current frame data for next iteration
            self.prev_boxes = pred_boxes.detach()
            self.prev_ids = gt_ids.detach()
            
            # Compute final Unitrack score (0-100)
            unitrack_score = (
                self.alpha_tracking * tracking_score +
                self.alpha_spatial * spatial_score +
                self.alpha_temporal * temporal_score
            ) * 100.0 / (self.alpha_tracking + self.alpha_spatial + self.alpha_temporal)
            
            # Convert to loss (higher Unitrack score = lower loss)
            losses[f'batch_{batch_idx}_unitrack'] = 1.0 - (unitrack_score / 100.0)
        
        # Average losses across batch
        if losses:
            losses['loss_unitrack'] = torch.stack([v for k, v in losses.items() if 'batch_' in k]).mean()
        else:
            losses['loss_unitrack'] = torch.tensor(0.0, device=device)
        
        return losses