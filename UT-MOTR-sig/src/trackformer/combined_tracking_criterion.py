import torch
import torch.nn as nn
from typing import Dict, List
import logging
from pprint import pformat

logger = logging.getLogger(__name__)

class CombinedTrackingCriterion(nn.Module):
    """Combines original tracking criterion with Unitrack criterion."""
    
    def __init__(self, tracking_criterion, unitrack_criterion, unitrack_weight=1.0, debug=True):
        super().__init__()
        self.tracking_criterion = tracking_criterion
        self.unitrack_criterion = unitrack_criterion
        self.unitrack_weight = unitrack_weight
        self.debug = debug
        
        # Keep original weight dict and add Unitrack weights
        self.weight_dict = tracking_criterion.weight_dict.copy()
        self.weight_dict['loss_unitrack'] = unitrack_weight

        # Initialize debug counters
        self.total_calls = 0
        self.calls_with_track_ids = 0
        
    def _debug_print_tensor_info(self, name: str, tensor: torch.Tensor):
        """Helper to print tensor debug info."""
        if isinstance(tensor, torch.Tensor):
            logger.info(f"{name}:")
            logger.info(f"  Shape: {tensor.shape}")
            logger.info(f"  Type: {tensor.dtype}")
            logger.info(f"  Device: {tensor.device}")
            if tensor.numel() > 0:
                logger.info(f"  Min/Max/Mean: {tensor.min():.4f}/{tensor.max():.4f}/{tensor.mean():.4f}")
                
    def _debug_loss_dict(self, prefix: str, loss_dict: Dict[str, torch.Tensor]):
        """Helper to print loss dict debug info."""
        logger.info(f"{prefix} Losses:")
        for k, v in loss_dict.items():
            logger.info(f"  {k}: {v.item():.4f}")
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss using both criteria.
        """
        self.total_calls += 1
        
        if self.debug:
            logger.info(f"\n{'='*80}\nForward Pass #{self.total_calls}")
            logger.info("Outputs keys: " + pformat(list(outputs.keys())))
            logger.info("First target keys: " + pformat(list(targets[0].keys())))
            
            # Debug tracking-related tensors
            if 'track_ids' in outputs:
                self._debug_print_tensor_info('track_ids', outputs['track_ids'])
            if 'pred_boxes' in outputs:
                self._debug_print_tensor_info('pred_boxes', outputs['pred_boxes'])
            
            # Debug target info
            for i, target in enumerate(targets):
                if 'track_ids' in target:
                    logger.info(f"Target {i} track_ids: {target['track_ids']}")
                if 'boxes' in target:
                    self._debug_print_tensor_info(f'Target {i} boxes', target['boxes'])

        # Get original tracking losses
        tracking_losses = self.tracking_criterion(outputs, targets)
        
        if self.debug:
            self._debug_loss_dict("Tracking", tracking_losses)
        
        # Get Unitrack losses only if we have track IDs
        if 'track_ids' in outputs:
            self.calls_with_track_ids += 1
            
            if self.debug:
                logger.info(f"\nProcessing Unitrack Loss (Call {self.calls_with_track_ids}/{self.total_calls})")
            
            try:
                unitrack_losses = self.unitrack_criterion(outputs, targets)
                
                if self.debug:
                    self._debug_loss_dict("Unitrack", unitrack_losses)
                
                # Scale Unitrack losses by weight
                unitrack_losses = {
                    k: v * self.unitrack_weight for k, v in unitrack_losses.items()
                }
                
                if self.debug:
                    self._debug_loss_dict("Weighted Unitrack", unitrack_losses)
                
                # Combine losses
                combined_losses = {**tracking_losses, **unitrack_losses}
                
                if self.debug:
                    logger.info("\nFinal Combined Losses:")
                    total_loss = sum(v for k, v in combined_losses.items() if k in self.weight_dict)
                    logger.info(f"Total Loss: {total_loss.item():.4f}")
                    
            except Exception as e:
                logger.error(f"Error computing Unitrack loss: {str(e)}")
                logger.error("Falling back to tracking losses only")
                combined_losses = tracking_losses
        else:
            if self.debug:
                logger.info("No track_ids found, skipping Unitrack loss")
            combined_losses = tracking_losses
            
        return combined_losses
        
    def train(self, mode=True):
        """Set both criteria to training mode."""
        self.tracking_criterion.train(mode)
        self.unitrack_criterion.train(mode)
        return super().train(mode)
        
    def eval(self):
        """Set both criteria to evaluation mode."""
        self.tracking_criterion.eval()
        self.unitrack_criterion.eval()
        return super().eval()
