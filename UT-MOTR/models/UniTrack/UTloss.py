import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Dict, Optional, Tuple, List
import pandas as pd
from collections import defaultdict


class Unitrack:
    """Unitrack metric incorporating traditional metrics with graph-based analysis."""
    
    def __init__(self, 
                 preds: Optional[List[Dict]] = None, 
                 gt_df: Optional[pd.DataFrame] = None, 
                 img_size: Tuple[int, int] = (1920, 1080),
                 iou_threshold: float = 0.5):
        """
        Initialize the Unitrack metric.

        Args:
            preds (Optional[List[Dict]]): List of prediction dictionaries.
            gt_df (Optional[pd.DataFrame]): Multi-indexed DataFrame for ground truth.
            img_size (Tuple[int, int], optional): Image size as (width, height). Defaults to (1920, 1080).
            iou_threshold (float, optional): IoU threshold for matching. Defaults to 0.5.
        """
        self.iou_threshold = iou_threshold
        self.img_diagonal = np.sqrt(img_size[0]**2 + img_size[1]**2)
        self.scale_factor = 1.0 / self.img_diagonal
        
        # Dynamic weighting parameters
        self.alpha_tracking = 2.0    # Tracking component weight
        self.alpha_spatial = 1.5     # Spatial consistency weight
        self.alpha_temporal = 1.8    # Temporal consistency weight
        self.beta_fp = 0.9           # False positive penalty
        self.beta_fn = 0.9           # False negative penalty
        self.gamma_switch = 1.5      # ID switch penalty
        
        # Load and preprocess data
        if preds is not None:
            self.pred_data = self._load_pred_data(preds)
        else:
            raise ValueError("Predictions data must be provided as a list of dictionaries.")
        
        self.has_gt = gt_df is not None
        if self.has_gt:
            self.gt_data = self._load_gt_data(gt_df)
        else:
            self.gt_data = None
        self._preprocess_data()
    
    def _load_pred_data(self, preds: List[Dict]) -> pd.DataFrame:
        """Load prediction data from a list of dictionaries."""
        try:
            df = pd.DataFrame(preds)
            required_columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y']
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing columns in predictions data: {missing_columns}")
            return df
        except Exception as e:
            raise ValueError(f"Error loading prediction data: {str(e)}")
    
    def _load_gt_data(self, gt_df: pd.DataFrame) -> pd.DataFrame:
        """Load ground truth data from a multi-indexed DataFrame."""
        try:
            if not isinstance(gt_df.index, pd.MultiIndex):
                raise ValueError("Ground truth DataFrame must have a MultiIndex with levels ['FrameId', 'Id'].")
            required_columns = ['X', 'Y', 'Width', 'Height', 'Confidence']
            missing_columns = set(required_columns) - set(gt_df.columns)
            if missing_columns:
                raise ValueError(f"Missing columns in ground truth data: {missing_columns}")
            
            # Reset index to have 'frame' and 'id' as columns
            df = gt_df.reset_index()
            df = df.rename(columns={
                'FrameId': 'frame',
                'Id': 'id',
                'X': 'bb_left',
                'Y': 'bb_top',
                'Width': 'bb_width',
                'Height': 'bb_height',
                'Confidence': 'conf'
            })
            # Ensure numerical types
            for col in ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            raise ValueError(f"Error loading ground truth data: {str(e)}")
    
    def _preprocess_data(self):
        """Enhanced data preprocessing with validation."""
        for df in [self.pred_data, self.gt_data] if self.has_gt else [self.pred_data]:
            if df is not None:
                df['frame'] = df['frame'].astype(int)
                df['id'] = df['id'].astype(int)
                df['area'] = df['bb_width'] * df['bb_height']
                df['center_x'] = df['bb_left'] + df['bb_width'] / 2
                df['center_y'] = df['bb_top'] + df['bb_height'] / 2
                
                # Validate data
                invalid_boxes = (df['bb_width'] <= 0) | (df['bb_height'] <= 0)
                if invalid_boxes.any():
                    print(f"Warning: Found {invalid_boxes.sum()} invalid boxes in frame {df['frame'].unique()}")
                    df.drop(index=df[invalid_boxes].index, inplace=True)
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Intersection over Union (IoU) between two bounding boxes."""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[0] + box1[2], box2[0] + box2[2])
        y2_min = min(box1[1] + box1[3], box2[1] + box2[3])
        
        inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _compute_tracking_error_no_gt(self, frame_data: pd.DataFrame) -> Dict:
        """Compute tracking quality metrics without ground truth."""
        if len(frame_data) < 2:
            # If there's only one or zero objects, some metrics become trivial
            return {
                'tracking_score': 0.0,
                'spatial_consistency': 1.0,
                'fp_rate': 0.0,
                'fn_rate': 0.0
            }
        
        # Mean detection confidence
        conf_scores = frame_data['conf'].values if 'conf' in frame_data.columns else np.ones(len(frame_data))
        mean_confidence = np.mean(conf_scores)
        
        # Compute average distance among centers
        centers = frame_data[['center_x', 'center_y']].values
        distances = cdist(centers, centers) * self.scale_factor
        np.fill_diagonal(distances, np.inf)
        
        min_distances = np.min(distances, axis=1)
        avg_min_distance = np.mean(min_distances)
        
        density_score = np.exp(-avg_min_distance * self.alpha_spatial)
        
        clustering_penalty = 1.0
        too_close = min_distances < 0.05
        if np.any(too_close):
            clustering_penalty = 1.0 - (np.sum(too_close) / len(frame_data)) * 0.5
        
        # Size consistency
        areas = frame_data['area'].values
        area_std = np.std(areas) / (np.mean(areas) + 1e-6)
        size_consistency = np.exp(-area_std)
        
        tracking_score = mean_confidence * density_score * clustering_penalty * size_consistency
        
        return {
            'tracking_score': tracking_score,
            'spatial_consistency': density_score * clustering_penalty,
            'fp_rate': 1.0 - mean_confidence,
            'fn_rate': 0.0
        }
    
    def _compute_assignment_metrics(self, gt_boxes: np.ndarray, pred_boxes: np.ndarray
                                   ) -> Tuple[float, int, int, int, List[int]]:
        """Compute assignment-based metrics (TP, FP, FN) using Hungarian algorithm."""
        num_gt = len(gt_boxes)
        num_pred = len(pred_boxes)
        
        if num_gt == 0 and num_pred == 0:
            return 0.0, 0, 0, 0, []
        if num_gt == 0:
            return 0.0, 0, num_pred, 0, []
        if num_pred == 0:
            return 0.0, num_gt, 0, 0, []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((num_gt, num_pred))
        for i in range(num_gt):
            for j in range(num_pred):
                iou_matrix[i, j] = self._compute_iou(gt_boxes[i], pred_boxes[j])
        
        # Hungarian algorithm for optimal assignment
        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Count matches above IoU threshold
        matches = [(i, j) for i, j in zip(row_ind, col_ind)
                   if iou_matrix[i, j] >= self.iou_threshold]
        tp = len(matches)
        fn = num_gt - tp
        fp = num_pred - tp
        
        precision = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        return precision, fn, fp, tp, col_ind.tolist()
    
    def compute_tracking_quality(self, 
                                 frame_data: pd.DataFrame, 
                                 gt_frame_data: Optional[pd.DataFrame]) -> Dict:
        """Compute tracking quality metrics for a single frame."""
        # If no ground truth, fallback to no-gt version
        if not self.has_gt or gt_frame_data is None:
            return self._compute_tracking_error_no_gt(frame_data)
        
        gt_boxes = gt_frame_data[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
        pred_boxes = frame_data[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
        
        precision, fn, fp, tp, assignments = self._compute_assignment_metrics(gt_boxes, pred_boxes)
        recall = tp / (tp + fn + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        # Spatial consistency penalty
        spatial_error = self._compute_spatial_consistency(frame_data)
        
        # Combine into a single tracking score with exponent penalty
        exponent = -self.beta_fp * fp / (len(pred_boxes) + 1e-6) \
                   -self.beta_fn * fn / (len(gt_boxes) + 1e-6)
        exponent = max(exponent, -20)  # Prevent extreme underflow
        tracking_score = (f1_score + 1e-6) * np.exp(exponent)
        
        return {
            'tracking_score': tracking_score,
            'spatial_consistency': 1 - spatial_error,
            'fp_rate': fp / (len(pred_boxes) + 1e-6),
            'fn_rate': fn / (len(gt_boxes) + 1e-6)
        }
    
    def _compute_spatial_consistency(self, frame_data: pd.DataFrame) -> float:
        """Compute spatial consistency using graph-based analysis within a frame."""
        if len(frame_data) < 2:
            return 0.0
            
        centers = frame_data[['center_x', 'center_y']].values
        distances = cdist(centers, centers) * self.scale_factor
        
        # Adjacency: whether two detections are "close" (<0.1 of image diagonal)
        adj_matrix = distances < 0.1
        np.fill_diagonal(adj_matrix, False)
        
        # Clustering coefficient measure
        clustering_coefs = []
        for i in range(len(frame_data)):
            neighbors = adj_matrix[i]
            if np.sum(neighbors) < 2:
                continue
            neighbor_indices = np.where(neighbors)[0]
            neighbor_subgraph = adj_matrix[np.ix_(neighbor_indices, neighbor_indices)]
            
            possible_connections = np.sum(neighbors) * (np.sum(neighbors) - 1) / 2
            actual_connections = np.sum(neighbor_subgraph) / 2
            if possible_connections > 0:
                clustering_coefs.append(actual_connections / possible_connections)
        
        return np.mean(clustering_coefs) if clustering_coefs else 0.0
    
    def compute_temporal_consistency(self, 
                                     prev_data: pd.DataFrame, 
                                     curr_data: pd.DataFrame) -> Dict:
        """Compute temporal consistency between two consecutive frames."""
        if len(prev_data) == 0 or len(curr_data) == 0:
            return {'temporal_score': 0.0, 'id_switches': 0, 'fragmentations': 0}
        
        prev_ids = set(prev_data['id'])
        curr_ids = set(curr_data['id'])
        maintained_ids = prev_ids.intersection(curr_ids)
        
        # Count ID switches and fragmentation
        id_switches = 0
        fragmentations = 0
        for track_id in maintained_ids:
            prev_box = prev_data[prev_data['id'] == track_id].iloc[0]
            curr_box = curr_data[curr_data['id'] == track_id].iloc[0]
            
            displacement = np.sqrt((prev_box['center_x'] - curr_box['center_x'])**2 +
                                   (prev_box['center_y'] - curr_box['center_y'])**2)
            
            # Large displacement => potential ID switch
            if displacement * self.scale_factor > 0.1:
                id_switches += 1
            
            # Large bounding box change => fragmentation
            if (abs(prev_box['bb_width'] - curr_box['bb_width']) 
                / (prev_box['bb_width'] + 1e-6) > 0.3 or 
                abs(prev_box['bb_height'] - curr_box['bb_height']) 
                / (prev_box['bb_height'] + 1e-6) > 0.3):
                fragmentations += 1
        
        # Score penalized by ID switches
        temporal_score = len(maintained_ids) / max(len(prev_ids), len(curr_ids))
        temporal_score *= np.exp(-self.gamma_switch 
                                 * (id_switches / (len(maintained_ids) + 1e-6)))
        
        return {
            'temporal_score': temporal_score,
            'id_switches': id_switches,
            'fragmentations': fragmentations
        }
    
    def compute(self) -> Dict:
        """
        Compute the overall Unitrack metric across all frames.

        Returns:
            Dict: A dictionary containing the final Unitrack score and its components.
        """
        frames = sorted(self.pred_data['frame'].unique())
        metrics = defaultdict(list)
        
        prev_frame_data = None
        
        for frame in frames:
            curr_frame_data = self.pred_data[self.pred_data['frame'] == frame]
            gt_frame_data = self.gt_data[self.gt_data['frame'] == frame] if self.has_gt else None
            
            # Per-frame tracking quality
            track_metrics = self.compute_tracking_quality(curr_frame_data, gt_frame_data)
            metrics['tracking_scores'].append(track_metrics['tracking_score'])
            metrics['spatial_scores'].append(track_metrics['spatial_consistency'])
            
            # Temporal consistency (if possible)
            if prev_frame_data is not None:
                temp_metrics = self.compute_temporal_consistency(prev_frame_data, curr_frame_data)
                metrics['temporal_scores'].append(temp_metrics['temporal_score'])
                metrics['id_switches'].append(temp_metrics['id_switches'])
                metrics['fragmentations'].append(temp_metrics['fragmentations'])
            
            prev_frame_data = curr_frame_data
        
        # Aggregate final results
        tracking_quality = np.mean(metrics['tracking_scores']) if metrics['tracking_scores'] else 0.0
        spatial_quality = np.mean(metrics['spatial_scores']) if metrics['spatial_scores'] else 0.0
        temporal_quality = np.mean(metrics['temporal_scores']) if metrics['temporal_scores'] else 1.0
        
        # Use inverse of std to weigh more stable components
        tracking_std = 1 / (np.std(metrics['tracking_scores']) + 1e-6)
        spatial_std = 1 / (np.std(metrics['spatial_scores']) + 1e-6)
        if len(metrics['temporal_scores']) > 1:
            temporal_std = 1 / (np.std(metrics['temporal_scores']) + 1e-6)
        else:
            temporal_std = 1.0
        
        total_weight = tracking_std + spatial_std + temporal_std
        w_track = tracking_std / total_weight
        w_spatial = spatial_std / total_weight
        w_temporal = temporal_std / total_weight
        
        # Final Unitrack score ~ 0..100
        unitrack_score = (w_track * tracking_quality + 
                       w_spatial * spatial_quality + 
                       w_temporal * temporal_quality) * 100
        
        return {
            'unitrack': unitrack_score,
            'components': {
                'tracking_quality': tracking_quality * 100,
                'spatial_quality': spatial_quality * 100,
                'temporal_quality': temporal_quality * 100,
                'tracking_weight': w_track,
                'spatial_weight': w_spatial,
                'temporal_weight': w_temporal,
                'avg_id_switches': np.mean(metrics['id_switches']) if metrics['id_switches'] else 0,
                'avg_fragmentations': np.mean(metrics['fragmentations']) if metrics['fragmentations'] else 0
            }
        }


class UnitrackLoss(Unitrack):
    """
    UnitrackLoss inherits from Unitrack and converts the Unitrack score into a loss function.
    Since Unitrack is higher-is-better, we can define a simple mapping to a lower-is-better loss.
    """
    def compute_loss(self) -> float:
        """
        Compute the Unitrack metric, then convert it to a loss.

        Returns:
            float: The Unitrack loss (lower is better).
        """
        unitrack_results = self.compute()
        unitrack_score = unitrack_results['unitrack']  # typically in [0, 100]
        
        # Option A: loss in [0, 1], where 1 = worst, 0 = best
        loss = 1.0 - (unitrack_score / 100.0)
        
        # Option B: negative unitrack (unbounded)
        # loss = -unitrack_score
        
        return loss


# ------------------------------------------------------------------------------
# Example usage:
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Suppose you have predictions in `preds` and ground truth DataFrame `gt_df`
    # Example (extremely simplified):
    preds = [
        {'frame': 1, 'id': 1, 'bb_left': 100, 'bb_top': 120, 'bb_width': 60, 'bb_height': 60, 'conf': 0.9, 'x': 0, 'y': 0},
        {'frame': 2, 'id': 1, 'bb_left': 100, 'bb_top': 120, 'bb_width': 60, 'bb_height': 60, 'conf': 0.8, 'x': 0, 'y': 0},
        # Add more predictions...
    ]
    
    # Fake ground truth as a MultiIndex DataFrame
    arrays = [
        [1, 1],  # FrameId
        [1, 2]   # Id
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=("FrameId", "Id"))
    gt_data = pd.DataFrame({
        "X": [100, 300],
        "Y": [120, 320],
        "Width": [50, 60],
        "Height": [60, 80],
        "Confidence": [1, 1]
    }, index=index)

    # Create and compute the Unitrack metric
    unitrack_metric = Unitrack(preds=preds, gt_df=gt_data)
    results = unitrack_metric.compute()
    print("Unitrack metric:", results)

    # Create and compute the UnitrackLoss
    unitrack_loss_obj = UnitrackLoss(preds=preds, gt_df=gt_data)
    loss_value = unitrack_loss_obj.compute_loss()
    print("Unitrack loss:", loss_value)
