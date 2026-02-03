import torch
import torch.nn as nn
from typing import Dict, List

class CombinedCriterion(nn.Module):
    """Combines the original tracking criterion with the Unitrack criterion."""
    
    def __init__(self, original_criterion, unitrack_criterion, unitrack_weight=1.0):
        super().__init__()
        self.original_criterion = original_criterion
        self.unitrack_criterion = unitrack_criterion
        self.unitrack_weight = unitrack_weight
        
        # Keep original weight dict but add Unitrack loss weight
        self.weight_dict = original_criterion.weight_dict.copy()
        self.weight_dict['loss_unitrack'] = unitrack_weight
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss from both criteria.
        
        Args:
            outputs: Dict containing model outputs
            targets: List of target dicts
            
        Returns:
            Dict containing all loss components
        """
        # Get original losses
        orig_losses = self.original_criterion(outputs, targets)
        
        # Get Unitrack losses
        unitrack_losses = self.unitrack_criterion(outputs, targets)
        
        # Combine all losses
        combined_losses = {**orig_losses, **unitrack_losses}
        
        return combined_losses
