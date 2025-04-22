import torch
import torch.nn as nn

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
    def __init__(self, cin=1, base=16, depth=4):
        super().__init__()
        downs, ups, skips = [], [], []
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
        orig_len = x.shape[-1]                
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = nn.functional.avg_pool1d(x, 2)

        x = self.bridge(x)

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)       
            skip = skips[-(i//2 + 1)]
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                x = nn.functional.pad(x, (0, diff))
            x = torch.cat([x, skip], dim=1)
            x = self.ups[i+1](x)     

        if x.shape[-1] != orig_len:
            x = nn.functional.interpolate(x, size=orig_len,mode='linear', align_corners=False)
        return self.head(x).squeeze(1)  