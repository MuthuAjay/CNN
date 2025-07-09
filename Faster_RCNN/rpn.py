import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np

class AnchorGenerator:
    """Generate anchor boxes for RPN"""
    
    def __init__(self, 
                 anchor_sizes: List[int] = [128, 256, 512],
                 aspect_ratios: List[float] = [0.5, 1.0, 2.0],
                 stride: int = 16):
        """
        Args:
            anchor_sizes: List of anchor box sizes
            aspect_ratios: List of aspect ratios for anchors
            stride: Stride of the feature map relative to input image
        """
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.stride = stride
        self.num_anchors = len(anchor_sizes) * len(aspect_ratios)
        
        # Generate base anchors
        self.base_anchors = self._generate_base_anchors()
    
    def _generate_base_anchors(self) -> torch.Tensor:
        """Generate base anchor boxes centered at (0, 0)"""
        anchors = []
        
        for size in self.anchor_sizes:
            for ratio in self.aspect_ratios:
                # Calculate width and height based on size and ratio
                h = size * np.sqrt(ratio)
                w = size / np.sqrt(ratio)
                
                # Create anchor box [x1, y1, x2, y2] centered at origin
                anchor = [-w/2, -h/2, w/2, h/2]
                anchors.append(anchor)
        
        return torch.tensor(anchors, dtype=torch.float32)
    
    def generate_anchors(self, feature_map_size: Tuple[int, int]) -> torch.Tensor:
        """Generate all anchor boxes for a feature map"""
        height, width = feature_map_size
        
        # Create grid of centers
        shift_x = torch.arange(0, width) * self.stride
        shift_y = torch.arange(0, height) * self.stride
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing='ij')
        
        # Flatten and create shifts
        shifts = torch.stack([shift_x.flatten(), shift_y.flatten(), 
                             shift_x.flatten(), shift_y.flatten()], dim=1)
        
        # Apply shifts to base anchors
        anchors = self.base_anchors.unsqueeze(0) + shifts.unsqueeze(1)
        anchors = anchors.view(-1, 4)
        
        return anchors

class RPNHead(nn.Module):
    """RPN head for classification and regression"""
    
    def __init__(self, 
                 in_channels: int,
                 num_anchors: int,
                 conv_channels: int = 512):
        """
        Args:
            in_channels: Number of input channels from backbone
            num_anchors: Number of anchors per location
            conv_channels: Number of channels in intermediate conv layer
        """
        super(RPNHead, self).__init__()
        
        self.num_anchors = num_anchors
        
        # 3x3 conv for feature transformation
        self.conv = nn.Conv2d(in_channels, conv_channels, kernel_size=3, 
                             stride=1, padding=1)
        
        # Classification head (objectness score)
        self.cls_logits = nn.Conv2d(conv_channels, num_anchors, 
                                   kernel_size=1, stride=1)
        
        # Regression head (bbox deltas)
        self.bbox_pred = nn.Conv2d(conv_channels, num_anchors * 4, 
                                  kernel_size=1, stride=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Feature map from backbone [B, C, H, W]
            
        Returns:
            cls_logits: Classification logits [B, num_anchors, H, W]
            bbox_pred: Bbox predictions [B, num_anchors*4, H, W]
        """
        # Apply 3x3 conv
        x = F.relu(self.conv(features))
        
        # Classification and regression heads
        cls_logits = self.cls_logits(x)
        bbox_pred = self.bbox_pred(x)
        
        return cls_logits, bbox_pred

class RPNLoss(nn.Module):
    """RPN loss computation"""
    
    def __init__(self, 
                 positive_threshold: float = 0.7,
                 negative_threshold: float = 0.3,
                 batch_size: int = 256,
                 positive_fraction: float = 0.5):
        """
        Args:
            positive_threshold: IoU threshold for positive anchors
            negative_threshold: IoU threshold for negative anchors
            batch_size: Number of anchors to sample for training
            positive_fraction: Fraction of positive anchors in batch
        """
        super(RPNLoss, self).__init__()
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.batch_size = batch_size
        self.positive_fraction = positive_fraction
        
        # Loss functions
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.bbox_loss = nn.SmoothL1Loss(reduction='none')
    
    def compute_iou(self, anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU between anchors and ground truth boxes"""
        # anchors: [N, 4], gt_boxes: [M, 4]
        # Returns: [N, M] IoU matrix
        
        anchors = anchors.unsqueeze(1)  # [N, 1, 4]
        gt_boxes = gt_boxes.unsqueeze(0)  # [1, M, 4]
        
        # Intersection
        lt = torch.max(anchors[:, :, :2], gt_boxes[:, :, :2])  # [N, M, 2]
        rb = torch.min(anchors[:, :, 2:], gt_boxes[:, :, 2:])  # [N, M, 2]
        
        wh = torch.clamp(rb - lt, min=0)  # [N, M, 2]
        intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
        
        # Areas
        anchor_area = (anchors[:, :, 2] - anchors[:, :, 0]) * (anchors[:, :, 3] - anchors[:, :, 1])
        gt_area = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0]) * (gt_boxes[:, :, 3] - gt_boxes[:, :, 1])
        
        # Union
        union = anchor_area + gt_area - intersection
        
        # IoU
        iou = intersection / (union + 1e-6)
        return iou
    
    def assign_targets(self, anchors: torch.Tensor, gt_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign classification and regression targets to anchors"""
        # Compute IoU
        iou_matrix = self.compute_iou(anchors, gt_boxes)  # [N, M]
        max_iou_per_anchor, matched_gt_idx = iou_matrix.max(dim=1)  # [N]
        
        # Initialize labels (-1: ignore, 0: negative, 1: positive)
        labels = torch.full((anchors.size(0),), -1, dtype=torch.long)
        
        # Assign negatives
        labels[max_iou_per_anchor < self.negative_threshold] = 0
        
        # Assign positives
        labels[max_iou_per_anchor >= self.positive_threshold] = 1
        
        # For each gt box, assign the anchor with highest IoU as positive
        if gt_boxes.size(0) > 0:
            max_iou_per_gt, matched_anchor_idx = iou_matrix.max(dim=0)
            labels[matched_anchor_idx] = 1
        
        # Regression targets (only for positive anchors)
        bbox_targets = torch.zeros_like(anchors)
        if gt_boxes.size(0) > 0:
            matched_gt_boxes = gt_boxes[matched_gt_idx]
            bbox_targets = self.encode_bbox_targets(anchors, matched_gt_boxes)
        
        return labels, bbox_targets
    
    def encode_bbox_targets(self, anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """Encode bbox targets as deltas"""
        # Anchor centers and sizes
        anchor_ctr_x = (anchors[:, 0] + anchors[:, 2]) / 2
        anchor_ctr_y = (anchors[:, 1] + anchors[:, 3]) / 2
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        
        # GT centers and sizes
        gt_ctr_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_ctr_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
        
        # Encode as deltas
        dx = (gt_ctr_x - anchor_ctr_x) / anchor_w
        dy = (gt_ctr_y - anchor_ctr_y) / anchor_h
        dw = torch.log(gt_w / anchor_w)
        dh = torch.log(gt_h / anchor_h)
        
        return torch.stack([dx, dy, dw, dh], dim=1)
    
    def sample_anchors(self, labels: torch.Tensor) -> torch.Tensor:
        """Sample anchors for training"""
        positive_mask = labels == 1
        negative_mask = labels == 0
        
        num_positive = positive_mask.sum().item()
        num_negative = negative_mask.sum().item()
        
        # Target number of positive and negative samples
        target_positive = min(int(self.batch_size * self.positive_fraction), num_positive)
        target_negative = min(self.batch_size - target_positive, num_negative)
        
        # Sample positive anchors
        if num_positive > target_positive:
            positive_indices = torch.where(positive_mask)[0]
            disable_indices = positive_indices[torch.randperm(int(num_positive))[int(target_positive):]]
            labels[disable_indices] = -1
        
        # Sample negative anchors
        if num_negative > target_negative:
            negative_indices = torch.where(negative_mask)[0]
            disable_indices = negative_indices[torch.randperm(int(num_negative))[int(target_negative):]]
            labels[disable_indices] = -1
        
        return labels
    
    def forward(self, 
                cls_logits: torch.Tensor,
                bbox_pred: torch.Tensor,
                anchors: torch.Tensor,
                gt_boxes: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            cls_logits: [B, num_anchors, H, W]
            bbox_pred: [B, num_anchors*4, H, W]
            anchors: [num_anchors_total, 4]
            gt_boxes: List of ground truth boxes for each image
        """
        batch_size = cls_logits.size(0)
        device = cls_logits.device
        
        # Reshape predictions
        cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(batch_size, -1)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        
        total_cls_loss = 0
        total_bbox_loss = 0
        
        for i in range(batch_size):
            # Assign targets
            labels, bbox_targets = self.assign_targets(anchors, gt_boxes[i])
            
            # Sample anchors
            labels = self.sample_anchors(labels)
            
            # Classification loss
            valid_mask = labels >= 0
            if valid_mask.sum() > 0:
                # Convert labels to float for binary classification
                binary_labels = (labels[valid_mask] == 1).float()
                cls_loss = self.cls_loss(
                    cls_logits[i][valid_mask],
                    binary_labels
                ).mean()
                total_cls_loss += cls_loss
            
            # Regression loss (only for positive anchors)
            positive_mask = labels == 1
            if positive_mask.sum() > 0:
                bbox_loss = self.bbox_loss(
                    bbox_pred[i][positive_mask],
                    bbox_targets[positive_mask]
                ).mean()
                total_bbox_loss += bbox_loss
        
        return {
            'cls_loss': torch.tensor(total_cls_loss / batch_size, device=cls_logits.device) if not isinstance(total_cls_loss, torch.Tensor) else total_cls_loss / batch_size,
            'bbox_loss': torch.tensor(total_bbox_loss / batch_size, device=cls_logits.device) if not isinstance(total_bbox_loss, torch.Tensor) else total_bbox_loss / batch_size,
            'total_loss': torch.tensor((total_cls_loss + total_bbox_loss) / batch_size, device=cls_logits.device) if not isinstance(total_cls_loss, torch.Tensor) else (total_cls_loss + total_bbox_loss) / batch_size
        }

class RegionProposalNetwork(nn.Module):
    """Complete Region Proposal Network"""
    
    def __init__(self,
                 in_channels: int,
                 anchor_sizes: List[int] = [128, 256, 512],
                 aspect_ratios: List[float] = [0.5, 1.0, 2.0],
                 stride: int = 16,
                 nms_threshold: float = 0.7,
                 pre_nms_top_n: int = 2000,
                 post_nms_top_n: int = 300,
                 score_threshold: float = 0.0):
        """
        Args:
            in_channels: Number of input channels from backbone
            anchor_sizes: List of anchor box sizes
            aspect_ratios: List of aspect ratios for anchors
            stride: Stride of the feature map relative to input image
            nms_threshold: NMS threshold for proposal filtering
            pre_nms_top_n: Number of top proposals before NMS
            post_nms_top_n: Number of top proposals after NMS
            score_threshold: Score threshold for filtering proposals
        """
        super(RegionProposalNetwork, self).__init__()
        
        self.stride = stride
        self.nms_threshold = nms_threshold
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.score_threshold = score_threshold
        
        # Components
        self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios, stride)
        self.rpn_head = RPNHead(in_channels, self.anchor_generator.num_anchors)
        self.loss_fn = RPNLoss()
    
    def decode_bbox_pred(self, bbox_pred: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode bbox predictions to actual coordinates"""
        # Anchor centers and sizes
        anchor_ctr_x = (anchors[:, 0] + anchors[:, 2]) / 2
        anchor_ctr_y = (anchors[:, 1] + anchors[:, 3]) / 2
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        
        # Decode predictions
        dx, dy, dw, dh = bbox_pred[:, 0], bbox_pred[:, 1], bbox_pred[:, 2], bbox_pred[:, 3]
        
        pred_ctr_x = dx * anchor_w + anchor_ctr_x
        pred_ctr_y = dy * anchor_h + anchor_ctr_y
        pred_w = torch.exp(dw) * anchor_w
        pred_h = torch.exp(dh) * anchor_h
        
        # Convert to coordinates
        pred_boxes = torch.zeros_like(bbox_pred)
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h  # y2
        
        return pred_boxes
    
    def filter_proposals(self, proposals: torch.Tensor, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply NMS and filtering to proposals"""
        # Sort by score
        _, sort_idx = scores.sort(descending=True)
        if len(sort_idx) > self.pre_nms_top_n:
            sort_idx = sort_idx[:self.pre_nms_top_n]
        
        proposals = proposals[sort_idx]
        scores = scores[sort_idx]
        
        # Apply NMS
        keep_idx = self.nms(proposals, scores, self.nms_threshold)
        
        # Keep top proposals
        if len(keep_idx) > self.post_nms_top_n:
            keep_idx = keep_idx[:self.post_nms_top_n]
        
        return proposals[keep_idx], scores[keep_idx]
    
    def nms(self, boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
        """Non-Maximum Suppression"""
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by scores
        _, order = scores.sort(descending=True)
        
        keep = []
        while order.numel() > 0:
            # Pick the box with highest score
            i = order[0]
            keep.append(i)
            
            if order.numel() == 1:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
            yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
            xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
            yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            intersection = w * h
            
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union
            
            # Keep boxes with IoU less than threshold
            inds = torch.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    def forward(self, 
                features: torch.Tensor,
                image_sizes: List[Tuple[int, int]],
                gt_boxes: Optional[List[torch.Tensor]] = None) -> Dict:
        """
        Args:
            features: Feature map from backbone [B, C, H, W]
            image_sizes: List of (height, width) for each image
            gt_boxes: Ground truth boxes for training
        """
        batch_size, _, feature_h, feature_w = features.shape
        
        # Generate anchors
        anchors = self.anchor_generator.generate_anchors((feature_h, feature_w))
        
        # RPN predictions
        cls_logits, bbox_pred = self.rpn_head(features)
        
        # Reshape predictions
        cls_logits_flat = cls_logits.permute(0, 2, 3, 1).reshape(batch_size, -1)
        bbox_pred_flat = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        
        # Convert to probabilities
        objectness_scores = torch.sigmoid(cls_logits_flat)
        
        # Generate proposals for each image
        proposals_list = []
        scores_list = []
        
        for i in range(batch_size):
            # Decode bbox predictions
            decoded_boxes = self.decode_bbox_pred(bbox_pred_flat[i], anchors)
            
            # Clip to image boundaries
            img_h, img_w = image_sizes[i]
            decoded_boxes[:, 0] = torch.clamp(decoded_boxes[:, 0], 0, img_w)
            decoded_boxes[:, 1] = torch.clamp(decoded_boxes[:, 1], 0, img_h)
            decoded_boxes[:, 2] = torch.clamp(decoded_boxes[:, 2], 0, img_w)
            decoded_boxes[:, 3] = torch.clamp(decoded_boxes[:, 3], 0, img_h)
            
            # Filter proposals
            proposals, scores = self.filter_proposals(decoded_boxes, objectness_scores[i])
            
            proposals_list.append(proposals)
            scores_list.append(scores)
        
        # Compute loss if training
        losses = {}
        if self.training and gt_boxes is not None:
            loss_dict = self.loss_fn(cls_logits, bbox_pred, anchors, gt_boxes)
            losses.update(loss_dict)
        
        return {
            'proposals': proposals_list,
            'scores': scores_list,
            'losses': losses
        }

# Example usage
if __name__ == "__main__":
    # Create RPN
    rpn = RegionProposalNetwork(
        in_channels=512,
        anchor_sizes=[128, 256, 512],
        aspect_ratios=[0.5, 1.0, 2.0]
    )
    
    # Example input
    batch_size = 2
    features = torch.randn(batch_size, 512, 50, 50)  # Feature map from backbone
    image_sizes = [(800, 800), (800, 800)]  # Image sizes
    
    # Forward pass
    rpn.eval()
    with torch.no_grad():
        outputs = rpn(features, image_sizes)
    
    print("RPN Outputs:")
    print(f"Number of proposals per image: {[len(p) for p in outputs['proposals']]}")
    print(f"Proposal shapes: {[p.shape for p in outputs['proposals']]}")
    print(f"Score shapes: {[s.shape for s in outputs['scores']]}")
    
    # Training example
    rpn.train()
    gt_boxes = [
        torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32),
        torch.tensor([[150, 150, 250, 250]], dtype=torch.float32)
    ]
    
    outputs = rpn(features, image_sizes, gt_boxes)
    print(f"\nTraining losses: {outputs['losses']}")