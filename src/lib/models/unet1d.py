import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader,random_split
import os, h5py

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {device}")

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=9, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=k//2),
            nn.BatchNorm1d(cout),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(cout, cout, k, padding=k//2),
            nn.BatchNorm1d(cout),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): 
        return self.net(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 8, 1),
            nn.BatchNorm1d(channels // 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        att = self.attention(x)
        return x * att

class NeuralPeakDetector(nn.Module):
    def __init__(self, cin=1, base=32, depth=4, dropout=0.1):
        super().__init__()
        
        # Encoder path with skip connections
        self.downs = nn.ModuleList()
        c = cin
        for d in range(depth):
            self.downs.append(nn.Sequential(
                ConvBlock(c, base*(2**d), dropout=dropout),
                AttentionBlock(base*(2**d))
            ))
            c = base*(2**d)
        
        # Bridge
        self.bridge = nn.Sequential(
            ConvBlock(c, c, dropout=dropout),
            AttentionBlock(c)
        )
        
        # Decoder path
        self.ups = nn.ModuleList()
        for d in reversed(range(depth-1)):
            self.ups.append(nn.ConvTranspose1d(c, c//2, 4, stride=2, padding=1))
            self.ups.append(nn.Sequential(
                ConvBlock(c + c//2, c//2, dropout=dropout),
                AttentionBlock(c//2)
            ))
            c = c // 2
        
        # Dual output heads:
        # 1. Signal reconstruction
        self.signal_head = nn.Conv1d(base, 1, 1)
        
        # 2. Peak confidence map (probability of peak at each position)
        self.peak_head = nn.Sequential(
            nn.Conv1d(base, base//2, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(base//2, 1, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        orig_len = x.shape[-1]
        
        # Store skip connections
        skips = []
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = nn.functional.avg_pool1d(x, 2)
        
        # Bridge
        x = self.bridge(x)
        
        # Decoder path
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[-(i//2 + 1)]
            
            # Handle size mismatches
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                x = nn.functional.pad(x, (0, diff))
                
            x = torch.cat([x, skip], dim=1)
            x = self.ups[i+1](x)
        
        # Ensure output has original length
        if x.shape[-1] != orig_len:
            x = nn.functional.interpolate(x, size=orig_len, mode='linear', align_corners=False)
        
        # Generate both outputs
        signal_output = self.signal_head(x).squeeze(1)
        peak_output = self.peak_head(x).squeeze(1)
        

        # recon = signal_output[0].detach().cpu().numpy()
        # pp = peak_output[0].detach().cpu().numpy()
        # plt.figure()
        # plt.plot(recon, label='Reconstruction')
        # plt.scatter(np.arange(1,168),peak_output[0].detach().cpu().numpy(),color='red',marker='o',s=80,label='Predicted Peaks')
        # plt.title("Reconstructed Signal with Predicted Peaks")
        # plt.legend()
        # plt.show()

        return {
            'signal': signal_output,
            'peak_map': peak_output
        }

# class DifferentiablePeakExtractor(nn.Module):
#     """
#     Neural differentiable peak extraction module
#     """
#     def __init__(self, threshold=0.5, min_distance=10):
#         super().__init__()
#         self.threshold = threshold
#         self.min_distance = min_distance
        
#     def forward(self, peak_map):
#         """
#         Extracts peaks from the peak probability map
        
#         Args:
#             peak_map: Tensor of shape [B, L] containing peak probabilities
            
#         Returns:
#             peak_indices: List of tensors containing peak indices for each batch
#             peak_values: List of tensors containing peak values for each batch
#         """
#         batch_size = peak_map.shape[0]
#         peak_indices = []
#         peak_values = []
        
#         for b in range(batch_size):
#             # 1. Threshold the peak map
#             above_threshold = (peak_map[b] > self.threshold).float()
            
#             # 2. Local maximum filtering (non-maximum suppression)
#             # Apply max pooling and check where values are preserved
#             padded = F.pad(peak_map[b].unsqueeze(0).unsqueeze(0), 
#                           (self.min_distance, self.min_distance), 
#                           mode='replicate')
            
#             max_pooled = F.max_pool1d(padded, kernel_size=2*self.min_distance+1, 
#                                      stride=1)
            
#             # Points that are both above threshold AND local maxima
#             is_peak = above_threshold * (peak_map[b] == max_pooled.squeeze()).float()
            
#             # Get indices and values
#             indices = torch.nonzero(is_peak).squeeze(-1)
#             values = peak_map[b][indices]
            
#             peak_indices.append(indices)
#             peak_values.append(values)
        
#         return peak_indices, peak_values

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class LearnablePeakExtractor(nn.Module):
    """
    Differentiable extractor with a trainable threshold.
    """
    def __init__(self, init_thresh=0.1, min_distance=2, sharpness=10.0):
        super().__init__()
        # store threshold in logit-space so it’s unconstrained
        self.logit_thresh = nn.Parameter(torch.logit(torch.tensor(init_thresh)))
        self.min_distance = min_distance
        self.sharpness    = sharpness

    def forward(self, peak_map):
        """
        Args:
            peak_map: FloatTensor of shape [B, L] in [0,1]
        Returns:
            peak_indices: List of 1D LongTensors, the selected peak positions per sample
            peak_values:  List of 1D FloatTensors, the corresponding probabilities
        """
        B, L = peak_map.shape

        # 1) compute actual threshold in (0,1)
        thresh = torch.sigmoid(self.logit_thresh)

        # 2) smooth gating around that threshold
        #    values near thresh get a value in (0,1), others near 0 or 1
        gate = torch.sigmoid(self.sharpness * (peak_map - thresh))  # [B, L]

        # 3) local‐max mask (soft)
        padded = F.pad(peak_map.unsqueeze(1), (self.min_distance,)*2, mode='replicate')
        pooled = F.max_pool1d(padded,kernel_size=2*self.min_distance+1,stride=1).squeeze(1)                # [B, L]
        local_mask = torch.sigmoid(self.sharpness * (peak_map - pooled))

        # 4) combine into a smooth “peakness” map
        smooth_peaks = peak_map * gate * local_mask              # [B, L]

        # 5) for discrete metrics, still extract hard indices at inference
        peak_indices = []
        peak_values  = []
        for b in range(B):
            # simple threshold on the smooth map
            mask_b = smooth_peaks[b] >= thresh
            idxs   = torch.nonzero(mask_b, as_tuple=False).squeeze(-1)
            peak_indices.append(idxs)
            peak_values.append(peak_map[b, idxs])

        return smooth_peaks, peak_indices, peak_values