import torch
import numpy as np
from tqdm import tqdm
import os, glob
import torch.nn.functional as F

from train.data_loader import get_dataloaders
from train.config import config
from lib.models.unet1d import NeuralPeakDetector, LearnablePeakExtractor
from lib.utils.misc_tools import epoch_num




def evaluate(checkpoint_path=None,config = None):
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    _, _, test_loader = get_dataloaders(config)

    model = NeuralPeakDetector(cin=1,base=config.model.BASE_CHANNELS,depth=config.model.MODEL_DEPTH,dropout=0.1).to(device)
    peak_extractor = LearnablePeakExtractor(threshold=0.5, min_distance=10)
    
    if checkpoint_path is None:
        # pick latest
        files = glob.glob(os.path.join(config.checkpoint.CKPT_DIR, 'ckpt_epoch_*.pth'))
        checkpoint_paths = sorted(files, key=epoch_num)
        checkpoint_path = checkpoint_paths[-1]
    data = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(data['model'])
    model.eval()

    all_preds, all_targets = [], []
    peak_accuracy_metrics = []
    with torch.no_grad():
        for x, y, peak_counts, *_ in tqdm(test_loader, desc='Test'):
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)

            # Calculate loss
            # loss, loss_components = peak_detection_loss(
            #     outputs, x, y
            # )
            
            # Extract peaks
            pred_peak_indices, _ = peak_extractor(outputs['peak_map'])
            
            # Calculate peak detection metrics
            for b in range(len(x)):
                # Get only the valid peaks (not padding)
                count = peak_counts[b].item()
                yt = y[b, :count].cpu().numpy()
                yp = pred_peak_indices[b].cpu().numpy()
                
                # Calculate metrics (precision, recall, F1)
                metrics = calculate_peak_metrics(
                    yt, 
                    yp,
                    tolerance=5
                )
                peak_accuracy_metrics.append(metrics)

    # f1 = f1_score(all_targets, all_preds)
    # print(f'Test F1 score: {f1:.4f}')
    # peak_metrics_matrix = []
    # metrics = calculate_peak_metrics(all_targets, all_preds, tolerance=1)

def calculate_peak_metrics(true_peaks, pred_peaks, tolerance=2):
    """
    Calculate precision, recall and F1 score for peak detection
    
    Args:
        true_peaks: Array of true peak indices
        pred_peaks: Array of predicted peak indices
        tolerance: How close predictions need to be to true peaks (in samples)
    """
    true_peaks = np.array(true_peaks)
    pred_peaks = np.array(pred_peaks)

    if len(true_peaks) == 0 and len(pred_peaks) == 0:
        return {'precision': 100.0, 'recall': 100.0, 'f1': 100.0}
    if len(true_peaks) == 0:
        return {'precision': 0.0, 'recall': 100.0, 'f1': 0.0}
    if len(pred_peaks) == 0:
        return {'precision': 100.0, 'recall': 0.0, 'f1': 0.0}

    # Compute absolute distance matrix: shape (num_true, num_pred)
    dist_matrix = np.abs(true_peaks[:, None] - pred_peaks[None, :])

    # Identify matches within tolerance
    match_matrix = dist_matrix <= tolerance

    # Greedy one-to-one assignment
    matched_true = set()
    matched_pred = set()

    # Loop over candidate (true_idx, pred_idx) pairs in order of increasing distance
    match_indices = np.argwhere(match_matrix)
    distances = dist_matrix[match_matrix]
    sorted_order = np.argsort(distances)
    for idx in sorted_order:
        true_idx, pred_idx = match_indices[idx]
        if true_idx not in matched_true and pred_idx not in matched_pred:
            matched_true.add(true_idx)
            matched_pred.add(pred_idx)

    tp = len(matched_pred)
    fp = len(pred_peaks) - tp
    fn = len(true_peaks) - len(matched_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision * 100, 'recall': recall * 100, 'f1': f1 * 100}


def peak_detection_loss(outputs,
                        y_mask,           # (B, L)  float32 in {0,1}
                        raw_signal,       # (B, 1, L) original PPG for recon
                        pos_weight=None,  # scalar weight for positives
                        peak_extractor = None,
                        alpha=0.5,
                        beta=0.5):
    """
    outputs: dict with
        'signal':    (B, L) reconstructed signal
        'peak_map':  (B, L) predicted probs in [0,1]
    y_mask:    (B, L) ground-truth binary mask
    raw_signal:(B, 1, L) the original PPG windows
    """

    # 1) Reconstruction loss (MSE)
    signal_loss = F.mse_loss(
        outputs['signal'],
        raw_signal.squeeze(1),
        reduction='mean'
    )
    if peak_extractor is not None:
        peaks,_,_ = peak_extractor(outputs['peak_map'])   # → Tensor (B, L)
    else:
        peaks = outputs['peak_map']
    
    # 2) Peak detection loss (BCE with optional pos_weight)
    if pos_weight is not None:
        # build a weight map: pos_weight on y=1, 1.0 on y=0
        weight = y_mask * pos_weight + (1.0 - y_mask)
        peak_loss = F.binary_cross_entropy(peaks,y_mask,weight=weight,reduction='mean')
        
    else:
        peak_loss = F.binary_cross_entropy(peaks,y_mask,reduction='mean')

    # 3) Combined
    total = alpha * peak_loss + beta * signal_loss

    return total, {
        'signal_loss': float(signal_loss),
        'peak_loss':   float(peak_loss),
        'total_loss':  float(total)
    }