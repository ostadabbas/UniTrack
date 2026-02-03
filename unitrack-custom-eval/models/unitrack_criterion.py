from typing import Dict, List
# import math
import torch
import torch.nn as nn
from torchvision.ops import box_iou 


class Unitrackrion(nn.Module):
    def __init__(self, img_size=(1920, 1080), iou_threshold=0.5):
        super().__init__()
        self.iou_threshold = iou_threshold
        # self.img_diagonal = math.sqrt(img_size[0]**2 + img_size[1]**2)
        self.scale_factor = 1.0 / 100.0  # Normalize by ~100 pixels
        self.register_buffer('alpha_tracking', torch.tensor(2.0))
        self.register_buffer('alpha_spatial', torch.tensor(1.0))
        self.register_buffer('alpha_temporal', torch.tensor(1.0))
        self.register_buffer('beta_fp', torch.tensor(0.9))
        self.register_buffer('beta_fn', torch.tensor(0.9))
        self.register_buffer('gamma_switch', torch.tensor(1.5))

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]) -> Dict[str, torch.Tensor]:
        device = outputs["pred_boxes"].device
        batch_size = len(targets)
        
        # Initialize loss components
        loss_tracking = torch.tensor(0.0, device=device)
        loss_spatial = torch.tensor(0.0, device=device)
        loss_temporal = torch.tensor(0.0, device=device)
        num_gt_tracks = 0  # Changed from num_tracks to num_gt_tracks for better normalization
        
        for batch_idx in range(batch_size):
            # Validate tensor shapes
            pred_boxes = outputs["pred_boxes"][batch_idx]
            assert pred_boxes.dim() == 2 and pred_boxes.size(1) == 4, \
                f"Pred boxes must be [N,4], got {pred_boxes.shape}"
                
            pred_logits = outputs["pred_logits"][batch_idx]
            pred_track_ids = outputs["track_ids"][batch_idx]
            
            # Filter valid tracks
            valid_track_mask = pred_track_ids >= 0
            if not valid_track_mask.any():
                continue
                
            pred_boxes = pred_boxes[valid_track_mask]
            pred_logits = pred_logits[valid_track_mask]
            pred_track_ids = pred_track_ids[valid_track_mask]
            
            # Get targets
            gt_boxes = targets[batch_idx]["boxes"]
            gt_track_ids = targets[batch_idx]["track_ids"]
            
            if len(gt_boxes) == 0:
                continue
                
            # Count unique GT tracks for normalization
            unique_gt_tracks = torch.unique(gt_track_ids[gt_track_ids >= 0])
            num_gt_tracks += len(unique_gt_tracks)
            
            # Vectorized IoU calculation
            ious = self._safe_box_iou(pred_boxes, gt_boxes)
            
            # Compute loss components
            loss_tracking += self._compute_tracking_loss(ious, pred_track_ids, gt_track_ids)
            loss_spatial += self._compute_spatial_consistency(pred_boxes, pred_track_ids)
            loss_temporal += self._compute_temporal_consistency(pred_boxes, pred_track_ids)
        
        # Normalize losses by number of GT tracks (not predicted tracks)
        if num_gt_tracks > 0:
            loss_tracking /= num_gt_tracks
            loss_spatial /= num_gt_tracks
            loss_temporal /= num_gt_tracks
        
        total_loss = (self.alpha_tracking * loss_tracking + 
                     self.alpha_spatial * loss_spatial + 
                     self.alpha_temporal * loss_temporal)
        
        # Debug logging (only in debug mode and rank 0)
        # if self.training and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f'Unitrack loss components - tracking: {loss_tracking:.6f}, '
        #                     f'spatial: {loss_spatial:.6f}, temporal: {loss_temporal:.6f}')
        # elif self.training and not torch.distributed.is_initialized():
        #     print(f'Unitrack loss components - tracking: {loss_tracking:.6f}, '
        #                     f'spatial: {loss_spatial:.6f}, temporal: {loss_temporal:.6f}')
        
        return {
            'loss_unitrack': total_loss,
            'loss_unitrack_tracking': loss_tracking,  # Return raw components without alpha scaling
            'loss_unitrack_spatial': loss_spatial,
            'loss_unitrack_temporal': loss_temporal
        }
    
    def _safe_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Vectorized IoU with shape validation and empty tensor handling"""
        # Validate input dimensions
        assert boxes1.dim() == 2 and boxes1.size(1) == 4, \
            f"boxes1 must be [N,4], got {boxes1.shape}"
        assert boxes2.dim() == 2 and boxes2.size(1) == 4, \
            f"boxes2 must be [M,4], got {boxes2.shape}"
        
        # Handle empty inputs
        if boxes1.size(0) == 0 or boxes2.size(0) == 0:
            return torch.zeros((boxes1.size(0), boxes2.size(0)), device=boxes1.device)
        
        # Use vectorized IoU implementation
        return box_iou(boxes1, boxes2)

    def _compute_tracking_loss(self, ious, pred_track_ids, gt_track_ids):
        """
        Fully vectorized tracking loss computation.
        ious: (num_pred, num_gt) IoU matrix
        pred_track_ids: (num_pred,) predicted track IDs
        gt_track_ids: (num_gt,) ground truth track IDs
        Returns: tracking loss
        """
        device = ious.device
        num_pred, num_gt = ious.shape
        
        if num_pred == 0 or num_gt == 0:
            return torch.tensor(0.0, device=device)
        
        # Create masks for valid track IDs
        valid_gt_mask = gt_track_ids >= 0
        valid_pred_mask = pred_track_ids >= 0
        
        if not valid_gt_mask.any() or not valid_pred_mask.any():
            return torch.tensor(0.0, device=device)
        
        # Filter to valid tracks only
        valid_gt_ids = gt_track_ids[valid_gt_mask]
        valid_pred_ids = pred_track_ids[valid_pred_mask]
        valid_ious = ious[valid_pred_mask][:, valid_gt_mask]
        
        # Get unique track IDs
        unique_gt_ids = torch.unique(valid_gt_ids)
        unique_pred_ids = torch.unique(valid_pred_ids)
        
        loss = torch.tensor(0.0, device=device)
        
        # === VECTORIZED APPROACH ===
        
        # 1. Create ID matching matrices
        # gt_id_matrix: (num_gt_valid, num_unique_gt) - one-hot encoding of GT IDs
        # pred_id_matrix: (num_pred_valid, num_unique_gt) - one-hot encoding of pred IDs for GT IDs
        gt_id_matrix = valid_gt_ids.unsqueeze(1) == unique_gt_ids.unsqueeze(0)  # (num_gt_valid, num_unique_gt)
        pred_id_matrix = valid_pred_ids.unsqueeze(1) == unique_gt_ids.unsqueeze(0)  # (num_pred_valid, num_unique_gt)
        
        # 2. For each GT track, compute losses
        for i, gt_id in enumerate(unique_gt_ids):
            gt_mask = gt_id_matrix[:, i]  # (num_gt_valid,) - mask for current GT track
            pred_mask = pred_id_matrix[:, i]  # (num_pred_valid,) - mask for predictions with same ID
            
            gt_indices = torch.where(gt_mask)[0]
            pred_indices = torch.where(pred_mask)[0]
            
            # False Negatives: GT objects with no matching prediction
            if len(pred_indices) == 0:
                # Use differentiable penalty based on number of GT instances
                loss = loss + self.beta_fn * len(gt_indices)
                continue
            
            # Localization penalty: penalize low IoU between matched predictions and GT
            track_ious = valid_ious[pred_indices][:, gt_indices]  # (num_pred_matched, num_gt_matched)
            
            # For each prediction, find its best GT match within this track
            if track_ious.numel() > 0:
                max_ious_per_pred, _ = track_ious.max(dim=1)  # (num_pred_matched,)
                localization_penalty = (1.0 - max_ious_per_pred).sum()
                loss = loss + localization_penalty
            
            # ID Switch penalty: penalize predictions with wrong ID that have high IoU with this GT track
            other_pred_mask = ~pred_mask  # Predictions with different IDs
            if other_pred_mask.any():
                other_pred_indices = torch.where(other_pred_mask)[0]
                other_ious = valid_ious[other_pred_indices][:, gt_indices]  # (num_other_pred, num_gt_matched)
                
                # Count ID switches: other predictions with high IoU to this GT track
                if other_ious.numel() > 0:
                    max_other_ious, _ = other_ious.max(dim=1)  # (num_other_pred,)
                    id_switches = (max_other_ious > self.iou_threshold).float()
                    switch_penalty = self.gamma_switch * id_switches.sum()
                    loss = loss + switch_penalty
        
        # 3. False Positives: predictions with IDs not in GT
        # Find predicted IDs that don't exist in GT
        gt_id_set = set(unique_gt_ids.cpu().tolist())
        for pred_id in unique_pred_ids:
            if pred_id.item() not in gt_id_set:
                # Count predictions with this non-existent ID
                fp_mask = valid_pred_ids == pred_id
                fp_count = fp_mask.sum().float()
                loss = loss + self.beta_fp * fp_count
        
        return loss
    
    def _compute_spatial_consistency(self, pred_boxes, pred_track_ids):
        """
        Vectorized spatial consistency loss with optional clustering term.
        Penalizes per-track width/height variability and spatial clustering.
        """
        device = pred_boxes.device
        n_boxes = pred_boxes.shape[0]
        if n_boxes <= 1:
            return torch.tensor(0.0, device=device)

        # 1. Box widths & heights
        widths = pred_boxes[:, 2] - pred_boxes[:, 0]   # (N,)
        heights = pred_boxes[:, 3] - pred_boxes[:, 1]

        # 2. Group info: unique track ids and per-box -> track mapping
        unique_ids, inverse, counts = torch.unique(
            pred_track_ids, return_inverse=True, return_counts=True
        )

        # 3. Per-track mean widths/heights using scatter operations
        width_sum = torch.zeros_like(unique_ids, dtype=widths.dtype)
        height_sum = torch.zeros_like(unique_ids, dtype=heights.dtype)
        width_sum.scatter_add_(0, inverse, widths)
        height_sum.scatter_add_(0, inverse, heights)

        mean_widths = width_sum / counts.float()
        mean_heights = height_sum / counts.float()

        # 4. Per-box deviation from its track's mean
        dev_w = (widths - mean_widths[inverse]).abs()
        dev_h = (heights - mean_heights[inverse]).abs()
        per_box_dev = dev_w + dev_h

        # 5. Average deviation per track
        dev_sum = torch.zeros_like(unique_ids, dtype=per_box_dev.dtype)
        dev_sum.scatter_add_(0, inverse, per_box_dev)
        track_loss = dev_sum / counts.float()

        # 6. Final loss = mean over tracks
        spatial_size_loss = track_loss.mean()

        # 7. Optional: Spatial clustering consistency
        # Compute box centers
        centers = torch.stack([
            (pred_boxes[:, 0] + pred_boxes[:, 2]) * 0.5,
            (pred_boxes[:, 1] + pred_boxes[:, 3]) * 0.5
        ], dim=1)  # (N, 2)

        # Apply scale factor for distance normalization
        centers_scaled = centers * self.scale_factor

        # Pairwise distances
        diffs = centers_scaled.unsqueeze(1) - centers_scaled.unsqueeze(0)  # (N, N, 2)
        distances = torch.norm(diffs, dim=2)  # (N, N)

        # Soft adjacency matrix (differentiable)
        k = 50.0  # Sharpness parameter
        distance_threshold = 0.1  # Normalized distance threshold
        adj_matrix = torch.sigmoid((distance_threshold - distances) * k)

        # Remove self-connections
        adj_matrix = adj_matrix - torch.diag_embed(torch.diagonal(adj_matrix))

        # Node degrees (weighted neighbor counts)
        neighbor_counts = adj_matrix.sum(dim=1)  # (N,)

        # Clustering coefficient computation
        valid_nodes = neighbor_counts >= 2.0
        cluster_loss = torch.tensor(0.0, device=device)
        
        if valid_nodes.any():
            # C_i = (A @ A ⊙ A).sum_j / (k_i * (k_i - 1) + ε)
            A = adj_matrix
            triple_mat = (A @ A) * A  # Element-wise product after matrix multiplication
            triangles = triple_mat.sum(dim=1)  # (N,)

            denom = neighbor_counts * (neighbor_counts - 1) + 1e-6
            clustering = torch.zeros_like(neighbor_counts)
            clustering[valid_nodes] = triangles[valid_nodes] / denom[valid_nodes]

            # Loss encourages high clustering (low loss for high clustering)
            cluster_loss = (1.0 - clustering[valid_nodes]).mean()

        # Combine spatial terms
        return spatial_size_loss + 0.1 * cluster_loss  # Weight clustering term lower
    
    def _compute_temporal_consistency(self, pred_boxes, pred_track_ids):
        """
        Vectorized temporal consistency loss.
        Penalizes acceleration magnitude for every track.
        Assumes input predictions for each track are sorted chronologically.
        """
        device = pred_boxes.device
        n = pred_boxes.shape[0]
        if n <= 2:
            return torch.tensor(0.0, device=device)

        # 1. Centers & track mapping
        centers = torch.stack([
            (pred_boxes[:, 0] + pred_boxes[:, 2]) * 0.5,
            (pred_boxes[:, 1] + pred_boxes[:, 3]) * 0.5
        ], dim=1)  # (N, 2)

        # Apply scale factor for consistent units
        centers = centers * self.scale_factor

        track_ids, inv, counts = torch.unique(
            pred_track_ids, return_inverse=True, return_counts=True
        )

        # Keep only tracks with ≥3 boxes for acceleration computation
        keep_mask = counts >= 3
        if keep_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Map track indices
        kept_tracks = track_ids[keep_mask]
        id_remap = torch.full_like(track_ids, -1, dtype=torch.long)
        id_remap[keep_mask] = torch.arange(keep_mask.sum(), device=device)
        track_idx_kept = id_remap[inv]

        # 2. Sort boxes within each track by original order
        sort_key = track_idx_kept * n + torch.arange(n, device=device)
        perm = sort_key.argsort()
        centers = centers[perm]
        track_idx_kept = track_idx_kept[perm]

        # 3. Compute velocities (consecutive frame differences within tracks)
        same_track = track_idx_kept[1:] == track_idx_kept[:-1]
        vel = centers[1:] - centers[:-1]  # (N-1, 2)
        vel = vel[same_track]
        vel_tracks = track_idx_kept[1:][same_track]

        # 4. Compute accelerations (consecutive velocity differences within tracks)
        if len(vel) <= 1:
            return torch.tensor(0.0, device=device)
            
        same_track_vel = vel_tracks[1:] == vel_tracks[:-1]
        acc = vel[1:] - vel[:-1]  # (M-1, 2)
        acc = acc[same_track_vel]
        acc_tracks = vel_tracks[1:][same_track_vel]

        if acc.shape[0] == 0:
            return torch.tensor(0.0, device=device)

        # 5. Per-track mean acceleration magnitude
        acc_norm = acc.norm(dim=1)

        num_kept = kept_tracks.numel()
        acc_sum = torch.zeros(num_kept, device=device)
        acc_count = torch.zeros(num_kept, device=device)

        acc_sum.scatter_add_(0, acc_tracks, acc_norm)
        acc_count.scatter_add_(0, acc_tracks, torch.ones_like(acc_norm))

        track_loss = acc_sum / acc_count.clamp(min=1.0)

        # 6. Mean over tracks
        return track_loss.mean()
