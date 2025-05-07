import glob, os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from train.config import config
from train.data_loader import get_dataloaders
from train.early_stopping import EarlyStoppingBasic
from train.evaluator import calculate_peak_metrics, peak_detection_loss
from train.visualize import plot_model_debug,create_training_video_from_frames

from lib.models.unet1d import NeuralPeakDetector,LearnablePeakExtractor

def find_latest_checkpoint(pattern=None):
    pattern = pattern or os.path.join(config.checkpoint.CKPT_DIR, 'ckpt_epoch_*.pth')
    files = glob.glob(pattern)
    if not files:
        return None
    # pick by epoch number
    epochs = [int(os.path.basename(f).split('_')[2]) for f in files]
    idx = epochs.index(max(epochs))
    return epochs[idx], files[idx]


def train(config):
    image_num = 0
    os.makedirs(config.plotting.save_dir, exist_ok=True)
    
    # Create directories for visualization
    frames_dir = os.path.join(config.plotting.save_dir, "training_frames")
    os.makedirs(frames_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'mps'
    tensorboard_path = os.path.join(config.plotting.save_dir,f"runs/exp/{config.checkpoint.MODEL_NAME}")
    writer = SummaryWriter(log_dir=tensorboard_path)
    
    model = NeuralPeakDetector(     cin=1,
        base=config.model.BASE_CHANNELS,
        depth=config.model.MODEL_DEPTH,
        dropout=0.1).to(device)
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'precision': [],
        'recall': [],
        'f1': []
    }


    # 1) Gather all labels from your training set (flattened)
    train_loader, val_loader,_ = get_dataloaders(config)
    if config.model.LossPosWeights is not None:
        pos_weight_val = config.model.LossPosWeights
    else:
        # 2) Compute class‐imbalance pos_weight from the *mask* yb
        counts = torch.zeros(2, dtype=torch.long, device=device)
        for _, yb, *_ in train_loader:
            # yb is now shape (B, win_len) of 0/1 floats
            flat = yb.view(-1).long().to(device)
            c    = torch.bincount(flat, minlength=2)  # c[0]=#zeros, c[1]=#ones
            counts += c
        neg, pos = counts[0].item(), counts[1].item()

        alpha = 0.5
        imbalance_ratio = neg / pos
        pos_weight_val  = imbalance_ratio ** alpha
        print(f"neg={neg}, pos={pos}")
        print(f"Imbalance ratio: {imbalance_ratio:.2f} → pos_weight={pos_weight_val:.2f}")

    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    # peak_extractor = DifferentiablePeakExtractor(threshold=0.5, min_distance=10)
    peak_extractor = LearnablePeakExtractor(init_thresh=0.4)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(peak_extractor.parameters()),lr=5e-4, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # optimizer, mode='max', patience=3, factor=0.8, threshold_mode='rel'
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    stopper = EarlyStoppingBasic(patience=5, min_delta=1e-4)


    demo_x, demo_y, demo_cnt = next(iter(val_loader))
    demo_x   = demo_x.to(device)
    demo_y = demo_y.to(device)
    demo_cnt = demo_cnt.to(device)

    start_epoch = 1
    if config.checkpoint.RESUME:
        ckpt = find_latest_checkpoint()
        if ckpt:
            start_epoch, path = ckpt
            data = torch.load(path, map_location=device)
            model.load_state_dict(data['model'])
            optimizer.load_state_dict(data['optim'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch += 1

    for epoch in range(start_epoch, config.EPOCHS):
        # Training
        model.train()
        train_losses = []
        for x, y,peak_counts  in tqdm(train_loader, desc=f'Train Epoch {epoch}/{config.EPOCHS}'):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss, loss_components = peak_detection_loss(outputs,y,x,peak_extractor = peak_extractor,pos_weight=pos_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss_components)

        # Validation
        model.eval()
        val_losses = []
        peak_accuracy_metrics = []
        with torch.no_grad():
            for x, y,peak_counts, in tqdm(val_loader, desc=f'Val Epoch {epoch}/{config.EPOCHS}'):
                x, y = x.to(device), y.to(device)

                outputs = model(x)
                loss, loss_components = peak_detection_loss(outputs,y,x,peak_extractor = peak_extractor,pos_weight=pos_weight)
                _, pred_peak_indices, _ = peak_extractor(outputs['peak_map'])
                
                # Calculate peak detection metrics
                for b in range(x.shape[0]):
                    true_idxs = np.nonzero(y[b].cpu().numpy() >= 0.5)[0]
                    pred_idxs = pred_peak_indices[b].cpu().numpy().astype(int)
                    m = calculate_peak_metrics(true_idxs, pred_idxs, tolerance=5)
                    peak_accuracy_metrics.append(m)
                val_losses.append(loss_components)

                frame_path = os.path.join(frames_dir, f"imageNum_{image_num}_epoch_{epoch:03d}.png")
                image_num = image_num+1
                logit = peak_extractor.logit_thresh.detach() 
                config.model.peak_threshold = torch.sigmoid(logit).item()
                if config.plotting.make_plots:
                    if image_num % 30 == 0:
                        plot_model_debug(
                            model,
                            peak_extractor = peak_extractor,
                            sample_batch       = demo_x,
                            sample_peaks       = demo_y,
                            sample_peak_counts = demo_cnt,
                            save_path          = frame_path,
                            epoch              = epoch,
                            return_fig         = config.plotting.return_fig,
                            config = config)
        
        scheduler.step()

        # Calculate epoch metrics
        avg_train_loss = np.mean([l['total_loss'] for l in train_losses])
        avg_val_loss = np.mean([l['total_loss'] for l in val_losses])
        avg_precision = np.mean([m['precision'] for m in peak_accuracy_metrics])
        avg_recall = np.mean([m['recall'] for m in peak_accuracy_metrics])
        avg_f1 = np.mean([m['f1'] for m in peak_accuracy_metrics])
        
        # Save metrics for plotting
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['precision'].append(avg_precision)
        training_history['recall'].append(avg_recall)
        training_history['f1'].append(avg_f1)

        print(f"Epoch {epoch}/{config.EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Peak Detection: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}")

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val',   avg_val_loss,   epoch)
        writer.add_scalar('F1/Val',      avg_f1,        epoch)
        writer.add_scalar('Precision',avg_precision,    epoch)
        writer.add_scalar('Recall',      avg_f1,        epoch)

        # Checkpoint
        if (epoch) % config.checkpoint.SAVE_EVERY == 0:
            torch.save({
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, os.path.join(config.checkpoint.CKPT_DIR, f'ckpt_epoch_{epoch}.pth'))
            # --- 1) generate & save a debug frame for this checkpoint
            if config.plotting.make_plots:
                # --- 2) stitch all frames into a video up to this checkpoint
                video_path = os.path.join(config.plotting.save_dir,f"training_progress_up_to_epoch_{epoch:03d}.mp4")
                create_training_video_from_frames(os.path.join(frames_dir, "imageNum_*_epoch_*.png"),video_path,fps=5)

        # Early stopping
        if stopper.step(avg_val_loss):
            print('Early stopping at epoch', epoch)
            break

    writer.close()
        
    # Plot training history
    plt.figure(figsize=(12, 10))
    
    # Loss plot
    plt.subplot(2, 1, 1)
    plt.plot(training_history['train_loss'], label='Train Loss')
    plt.plot(training_history['val_loss'], label='Val Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Metrics plot
    plt.subplot(2, 1, 2)
    plt.plot(training_history['precision'], label='Precision')
    plt.plot(training_history['recall'], label='Recall')
    plt.plot(training_history['f1'], label='F1 Score')
    plt.title('Peak Detection Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(config.plotting.save_dir, 'training_history.png'))
    plt.show()

    return config