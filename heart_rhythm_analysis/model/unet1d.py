import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import io
import imageio
import math

def compute_grid(n_plots, aspect=1.5):
    # guesses cols ≈ sqrt(n_plots * aspect)
    cols = min(n_plots, math.ceil(math.sqrt(n_plots * aspect)))
    rows = math.ceil(n_plots / cols)
    return rows, cols

# ----------------------------- model -------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=9, s=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, stride=s, padding=k//2),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, k, stride=1, padding=k//2),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x): return self.net(x)

class UNet1D(nn.Module):
    def __init__(self, cin=1, base=16, depth=2, bDebug=False, bVideo=False,ref_y = None):
        """
        bDebug: if True, debug plots will be generated.
        bVideo: if True, instead of plotting, debug plots are saved as frames for a video.
        """
        super().__init__()
        self.bDebug = bDebug
        self.bVideo = bVideo
        self.debug_frames = [] if bVideo else None
        self.ref_y = ref_y
        
        self.debug_interval = 10      # do debug every 10 calls
        self._debug_counter = 0
        downs, ups = [], []
        c = cin
        for d in range(depth):
            downs.append(ConvBlock(c, base*(2**d), k=9))
            c = base*(2**d)
        self.downs = nn.ModuleList(downs)

        self.bridge = ConvBlock(c, c, k=9)

        for d in reversed(range(depth-1)):
            ups.append(nn.ConvTranspose1d(c, c//2, 4, stride=2, padding=1))
            ups.append(ConvBlock(c + c//2, c//2, k=9))
            c = c // 2
        self.ups = nn.ModuleList(ups)

        self.head = nn.Conv1d(base, 1, 1)

    def forward(self, x):
        
            # increment & check
        self._debug_counter += 1
        do_debug = self.bDebug and (self._debug_counter % self.debug_interval == 0)

        # Store original input for plotting
        orig_x = x.clone()
        orig_len = x.shape[-1]

        if do_debug:
            debug_plots = []  # Collect debug plotting info
            input_np  = orig_x[0, 0].detach().cpu().numpy()
            curr_y = self.ref_y
            debug_plots.append({
                'plot_type': 'overlay',
                'data':       input_np,
                'markers': {
                    'original': {'x': curr_y, 'y': input_np[curr_y]},
                    },
                'title': "Input Signal",
                'xlabel': "Time steps",
                'ylabel': "Amplitude"
            })

        # Encoder path with skip connections
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x)
            skips.append(x)
            
            # Debug plot: After each down block
            if do_debug:
                # For each down block, plot first channel of first feature map
                feature_map = x[0, 0].detach().cpu().numpy()
                debug_plots.append({
                    'plot_type': 'line',
                    'data': feature_map,
                    'title': f"After Down Block {i+1}",
                    'xlabel': "Time steps",
                    'ylabel': "Feature Activation"
                })
                
                # Also plot a heatmap of all channels for this sample
                heatmap_data = x[0].detach().cpu().numpy()
                debug_plots.append({
                    'plot_type': 'heatmap',
                    'data': heatmap_data,
                    'title': f"Feature Maps After Down Block {i+1}",
                    'xlabel': "Time steps",
                    'ylabel': "Channel"
                })
            
            x = nn.functional.avg_pool1d(x, 2)
            
            # Debug plot: After pooling
            if do_debug:
                debug_plots.append({
                    'plot_type': 'line',
                    'data': x[0, 0].detach().cpu().numpy(),
                    'title': f"After Pooling {i+1}",
                    'xlabel': "Time steps",
                    'ylabel': "Feature Activation"
                })

        # Bridge
        x = self.bridge(x)
        
        # Debug plot: After bridge
        if do_debug:
            debug_plots.append({
                'plot_type': 'line',
                'data': x[0, 0].detach().cpu().numpy(),
                'title': "After Bridge",
                'xlabel': "Time steps",
                'ylabel': "Feature Activation"
            })

        # Decoder path
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            
            # Debug plot: After up-convolution
            if do_debug:
                debug_plots.append({
                    'plot_type': 'line',
                    'data': x[0, 0].detach().cpu().numpy(),
                    'title': f"After Up-Conv {i//2 + 1}",
                    'xlabel': "Time steps",
                    'ylabel': "Feature Activation"
                })
            
            skip = skips[-(i//2 + 1)]
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                x = nn.functional.pad(x, (0, diff))
            x = torch.cat([x, skip], dim=1)
            
            # Debug plot: After concatenation
            if do_debug:
                debug_plots.append({
                    'plot_type': 'line',
                    'data': x[0, 0].detach().cpu().numpy(),
                    'title': f"After Concat {i//2 + 1}",
                    'xlabel': "Time steps",
                    'ylabel': "Feature Activation"
                })
            
            x = self.ups[i+1](x)
            
            # Debug plot: After up block
            if do_debug:
                debug_plots.append({
                    'plot_type': 'line',
                    'data': x[0, 0].detach().cpu().numpy(),
                    'title': f"After Up Block {i//2 + 1}",
                    'xlabel': "Time steps",
                    'ylabel': "Feature Activation"
                })

        # Final interpolation if needed
        if x.shape[-1] != orig_len:
            x = nn.functional.interpolate(x, size=orig_len, mode='linear', align_corners=False)
            
            # Debug plot: After final interpolation
            if do_debug:
                debug_plots.append({
                    'plot_type': 'line',
                    'data': x[0, 0].detach().cpu().numpy(),
                    'title': "After Final Interpolation",
                    'xlabel': "Time steps",
                    'ylabel': "Feature Activation"
                })

        # Final output
        output = self.head(x).squeeze(1)
        
        # Debug plot: Final output
        if do_debug:
            debug_plots.append({
                'plot_type': 'line',
                'data': output[0].detach().cpu().numpy(),
                'title': "Final Output",
                'xlabel': "Time steps",
                'ylabel': "Amplitude"
            })
            
            # Overlay: Input vs Output
            input_signal = orig_x[0, 0].detach().cpu().numpy()
            output_signal = output[0].detach().cpu().numpy()
            debug_plots.append({
                'plot_type': 'multiple_line',
                'data': {'Input': input_signal, 'Output': output_signal},
                'title': "Input vs Output",
                'xlabel': "Time steps",
                'ylabel': "Amplitude"
            })
            
            # Find peaks in the output (simple threshold-based detection)
            # output_np = output[0].detach().cpu().numpy()
            # input_np  = orig_x[0, 0].detach().cpu().numpy()
            # threshold = 0.5 * output_np.max()   # move this out of the loop
            # peak_indices = []
            # peak_values = []
            # orig_peak_indices = []
            # orig_peak_values  = []
            # # Simple peak detection (could be improved with scipy.signal.find_peaks)
            # window_size = 10
            # for i in range(window_size, len(output_np) - window_size):
            #     window = output_np[i-window_size:i+window_size+1]
            #     if output_np[i] == window.max() and output_np[i] > threshold:
            #         peak_indices.append(i)
            #         peak_values.append(output_np[i])

            #         orig_peak_indices.append(i)
            #         orig_peak_values.append(input_np[i])

            # 2) Pick a threshold (e.g. 0.5)
            thresh = 0.5

            # 3) Find peaks on the probability curve
            #    SciPy does C-speed peak finding with height + min distance
            # 1) pick your threshold
            threshold = 0.5 * output_signal.max()   # or any constant T

            # 2) build the mask
            mask = output_signal > threshold        # 1D boolean array, same length as output_np

            # 3) extract indices & values
            peak_indices = np.nonzero(mask)[0]  # all i where output_np[i] > threshold
            peak_values  = output_signal[peak_indices]

            # Overlay peaks on output
            debug_plots.append({
                'plot_type': 'overlay',
                'data':       output_signal,
                'markers': {
                    'detected': {'x': peak_indices, 'y': peak_values},
                    'original': {'x': curr_y, 'y': output_signal[curr_y]}
                },
                'title': 'Output Signal with Detected (red) & Original (blue) Peaks',
                'xlabel': 'Sample index',
                'ylabel': 'Amplitude'
            })
            debug_plots[0]['markers']['detected'] = {'x': peak_indices, 'y': peak_values}

        # Render all debug plots
        if do_debug:
            n_plots = len(debug_plots)
            rows, cols = compute_grid(n_plots)
            fig = plt.figure(figsize=(cols * 5, rows * 2))
            
            for i, plot_info in enumerate(debug_plots):
                ax = fig.add_subplot(rows, cols, i + 1)
                
                if plot_info['plot_type'] == 'line':
                    ax.plot(plot_info['data'])
                    ax.set_xlim(0, len(plot_info['data']))
                
                elif plot_info['plot_type'] == 'multiple_line':
                    data_dict = plot_info['data']
                    colors = ['b', 'r', 'g', 'c', 'm']
                    for j, (key, values) in enumerate(data_dict.items()):
                        ax.plot(values, color=colors[j % len(colors)], label=key)
                    ax.legend()
                
                elif plot_info['plot_type'] == 'overlay':
                    ax.plot(plot_info['data'], label='Signal')
                    m = plot_info['markers']
                    # detected peaks in red circles
                    ax.scatter(m['detected']['x'], m['detected']['y'],
                            color='red', marker='o', label='Detected Peaks')
                    # original peaks in blue crosses
                    ax.scatter(m['original']['x'], m['original']['y'],
                            color='blue', marker='x', label='Original Peaks')
                    ax.legend()
                
                elif plot_info['plot_type'] == 'heatmap':
                    im = ax.imshow(plot_info['data'], aspect='auto', cmap='viridis')
                    plt.colorbar(im, ax=ax)
                
                ax.set_title(plot_info['title'])
                ax.set_xlabel(plot_info['xlabel'])
                ax.set_ylabel(plot_info['ylabel'])
            
            plt.tight_layout()
            
            if self.bVideo:
                # Save figure to an image buffer for video
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                frame = imageio.imread(buf)
                self.debug_frames.append(frame)
                buf.close()
                plt.close(fig)
            else:
                plt.show()

        return output

    def save_debug_video(self, filename="unet_debug_video.mp4", fps=2):
        """If bVideo is enabled and debug frames have been accumulated,
        convert the frames into a video saved to the given filename."""
        if self.bVideo and self.debug_frames:
            imageio.mimwrite(filename, self.debug_frames, fps=fps)
            print(f"Saved video with {len(self.debug_frames)} frames to {filename}")
            # Optionally clear the saved frames after saving:
            # self.debug_frames = []

# Example usage:
if __name__ == "__main__":
    model = UNet1D(cin=1, base=16, depth=4, bDebug=True)
    x = torch.randn(1, 1, 1024)
    output = model(x)