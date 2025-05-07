import matplotlib.pyplot as plt
import torch
from lib.models.unet1d import LearnablePeakExtractor
import numpy as np
from train.evaluator import calculate_peak_metrics
import os


def plot_peaks(t, signal, gt_peaks, pred_peaks, title=None):
    plt.figure()
    plt.plot(t, signal, label='PPG')
    plt.scatter(t[gt_peaks], signal[gt_peaks], marker='^', label='GT Peaks')
    plt.scatter(t[pred_peaks], signal[pred_peaks], marker='x', label='Pred Peaks')
    if title:
        plt.title(title)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

def plot_model_debug(model, sample_batch, sample_peaks=None, sample_peak_counts=None, 
                     save_path=None, epoch=None, return_fig=False,config=None):
    """
    Create a detailed visualization of the model's performance on a batch
    
    Args:
        model: The trained model
        sample_batch: Tensor of shape (B,1,L) containing PPG windows
        sample_peaks: Either a tensor of shape (B,K) with padded peak indices or a mask (B,L)
        sample_peak_counts: Tensor of shape (B,) giving valid peaks per sample, or None
        save_path: Path to save the figure
        epoch: Current epoch number (for title)
        return_fig: If True, returns the figure object
    """
    model.eval()
    peak_extractor = LearnablePeakExtractor()
    
    if config is not None:
        peak_prob_th = config.model.peak_threshold

    with torch.no_grad():
        outputs = model(sample_batch)  # {'signal': (B,L), 'peak_map': (B,L)}
        _, pred_peak_indices, pred_peak_values = peak_extractor(outputs['peak_map'])
        
        fig = plt.figure(figsize=(15, 10))
        b = np.random.randint(31)  # visualize first sample

        # Panel 1: Input signal
        ax1 = fig.add_subplot(3, 1, 1)
        signal = sample_batch[b, 0].cpu().numpy()
        ax1.plot(signal, label='Input')
        # Overlay true peaks (derive indices straight from the mask)
        if sample_peaks is not None:
            mask = sample_peaks[b].cpu().numpy()
            true_peaks = np.nonzero(mask >= 0.5)[0]   # integer indices where mask == 1
            ax1.scatter(true_peaks,signal[true_peaks],color='blue',marker='x',s=100,label='True Peaks')
        title = "Input Signal"
        if epoch is not None:
            title += f" (Epoch {epoch})"
        ax1.set_title(title)
        ax1.legend()

        # Panel 2: Reconstruction
        ax2 = fig.add_subplot(3, 1, 2)
        recon = outputs['signal'][b].cpu().numpy()
        ax2.plot(recon, label='Reconstruction')
        ax2.set_title("Reconstructed Signal")
        ax2.legend()

        pm = outputs['peak_map'][b].cpu().numpy()
        pp = pred_peak_indices[b].cpu().numpy().astype(int)
        pv = pred_peak_values[b].cpu().numpy()
        # Panel 3: Peak probability map
        ax3 = fig.add_subplot(3, 1, 3)
        
        ax3.plot(pm, label='Peak Map')
        # show predicted peaks

        ax3.scatter(pp, pv, color='red', marker='o', s=100, label='Predicted Peaks')
        ax3.scatter(true_peaks,pm[true_peaks],color='blue',marker='x',s=100,label='True Peaks')
        ax3.axhline(peak_prob_th, color='g', linestyle='--',linewidth=2.0, label='Threshold')
        ax3.set_title(f"Peak Probability Map with Detected Peaks | Th at {peak_prob_th}")
        ax3.legend()

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close(fig)
        if return_fig:
            return fig
        else:
            plt.show()
            plt.close(fig)


def create_training_video_from_frames(frames_pattern, output_path, fps=2):
    """
    Creates a video from a sequence of image files
    
    Args:
        frames_pattern: Glob pattern for frame images (e.g., "frames/epoch_*.png")
        output_path: Path to save the video file
        fps: Frames per second in the output video
    """
    import glob
    import imageio
    
    # Get all frame files and sort them
    frame_files = sorted(glob.glob(frames_pattern))
    
    if not frame_files:
        print(f"No frames found matching pattern: {frames_pattern}")
        return
    
    print(f"Creating video from {len(frame_files)} frames...")
    
    # Read frames
    frames = [imageio.imread(file) for file in frame_files]
    
    # Create video
    imageio.mimsave(output_path, frames, fps=fps)
    
    print(f"Video saved to: {output_path}")


def visualize_peak_detection(model, signal_batch, peak_positions=None, peak_counts=None, save_dir=None):
    """
    Visualize peak detection performance on multiple samples
    
    Args:
        model: The trained model
        signal_batch: Batch of signals to visualize
        peak_positions: Ground truth peak positions (with padding)
        peak_counts: Number of actual peaks per sample
        save_dir: Directory to save visualizations
    
    Returns:
        Dictionary with average metrics if ground truth is provided
    """
    model.eval()
    peak_extractor = DifferentiablePeakExtractor(threshold=0.5, min_distance=10)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(signal_batch)
        
        # Extract peaks
        pred_peak_indices, pred_peak_values = peak_extractor(outputs['peak_map'])
        
        # Visualize
        for b in range(len(signal_batch)):
            plt.figure(figsize=(15, 10))
            
            # Original signal
            plt.subplot(3, 1, 1)
            signal = signal_batch[b, 0].cpu().numpy()
            plt.plot(signal)
            
            # Add true peaks if provided
            if peak_positions is not None:
                if peak_counts is not None:
                    # Only use valid peaks (not padding)
                    count = peak_counts[b].item()
                    true_peaks = peak_positions[b, :count].cpu().numpy()
                else:
                    # Filter out padding values (-1)
                    true_peaks = peak_positions[b].cpu().numpy()
                    true_peaks = true_peaks[true_peaks >= 0]
                
                plt.scatter(true_peaks, signal[true_peaks], color='blue', marker='x', s=100, 
                        label='True Peaks')
            
            plt.title("Input Signal")
            plt.legend()
            
            # Reconstructed signal
            plt.subplot(3, 1, 2)
            recon_signal = outputs['signal'][b].cpu().numpy()
            plt.plot(recon_signal)
            plt.title("Reconstructed Signal")
            
            # Peak probability map with detected peaks
            plt.subplot(3, 1, 3)
            peak_map = outputs['peak_map'][b].cpu().numpy()
            plt.plot(peak_map)
            
            # Show predicted peaks
            pred_peaks = pred_peak_indices[b].cpu().numpy()
            pred_values = pred_peak_values[b].cpu().numpy()
            plt.scatter(pred_peaks, pred_values, color='red', marker='o', s=100,
                    label='Predicted Peaks')
            
            # Show threshold
            plt.axhline(y=0.5, color='g', linestyle='--', label='Threshold')
            plt.title("Peak Probability Map with Detected Peaks")
            plt.legend()
            
            plt.tight_layout()
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'sample_{b}.png'))
            
            plt.show()
        
        # If we have true peak positions, calculate metrics
        if peak_positions is not None:
            all_metrics = []
            
            for b in range(len(signal_batch)):
                if peak_counts is not None:
                    # Only use valid peaks (not padding)
                    count = peak_counts[b].item()
                    true_peaks = peak_positions[b, :count].cpu().numpy()
                else:
                    # Filter out padding values (-1)
                    true_peaks = peak_positions[b].cpu().numpy()
                    true_peaks = true_peaks[true_peaks >= 0]
                
                pred_peaks = pred_peak_indices[b].cpu().numpy()
                
                metrics = calculate_peak_metrics(true_peaks, pred_peaks, tolerance=5)
                all_metrics.append(metrics)
                
                print(f"Sample {b}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1 Score: {metrics['f1']:.4f}")
            
            # Calculate average metrics
            avg_precision = np.mean([m['precision'] for m in all_metrics])
            avg_recall = np.mean([m['recall'] for m in all_metrics])
            avg_f1 = np.mean([m['f1'] for m in all_metrics])
            
            print("\nAverage Metrics:")
            print(f"  Precision: {avg_precision:.4f}")
            print(f"  Recall: {avg_recall:.4f}")
            print(f"  F1 Score: {avg_f1:.4f}")
            
            return {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1
            }