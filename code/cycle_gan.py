import os
import bisect
import numpy as np
import pandas as pd
from PIL import Image
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


class DAVIS240CTestDataset(Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.height, self.width = 180, 240

        images_txt = os.path.join(base_dir, "images.txt")
        self.image_meta = []
        with open(images_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.image_meta.append((float(parts[0]), parts[1]))

        events_txt = os.path.join(base_dir, "events.txt")
        print(f"Loading {events_txt}...")
        df_events = pd.read_csv(
            events_txt,
            sep=r'\s+',
            header=None,
            names=['t', 'x', 'y', 'p'],
            dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'p': np.int8}
        )
        self.event_timestamps = df_events['t'].values
        self.event_x = df_events['x'].values
        self.event_y = df_events['y'].values
        self.event_p = df_events['p'].values
        del df_events

    def __len__(self):
        return len(self.image_meta) - 1

    def __getitem__(self, idx):
        t0, img_rel_path0 = self.image_meta[idx]
        t1, img_rel_path1 = self.image_meta[idx + 1]

        path0 = os.path.join(self.base_dir, img_rel_path0)
        path1 = os.path.join(self.base_dir, img_rel_path1)

        img0 = TF.to_tensor(Image.open(path0).convert('L'))
        img1 = TF.to_tensor(Image.open(path1).convert('L'))

        idx_start = bisect.bisect_left(self.event_timestamps, t0)
        idx_end = bisect.bisect_right(self.event_timestamps, t1)

        xs = self.event_x[idx_start:idx_end]
        ys = self.event_y[idx_start:idx_end]
        ps = self.event_p[idx_start:idx_end]

        event_frame = np.zeros((2, self.height, self.width), dtype=np.float32)
        valid = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
        xs, ys, ps = xs[valid], ys[valid], ps[valid]

        np.add.at(event_frame[0], (ys[ps > 0], xs[ps > 0]), 1.0)
        np.add.at(event_frame[1], (ys[ps <= 0], xs[ps <= 0]), 1.0)

        # Log dynamic range compression matching training
        event_frame = np.log1p(event_frame)
        if event_frame.max() > 0:
            event_frame = event_frame / event_frame.max()

        return img0, img1, torch.from_numpy(event_frame)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class ResNetGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, num_res_blocks=6):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        curr_dim = 64
        for _ in range(2):
            layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim *= 2

        for _ in range(num_res_blocks):
            layers += [ResidualBlock(curr_dim)]

        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim //= 2

        layers += [
            nn.Conv2d(64, out_channels, kernel_size=7, padding=3),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def compute_ssim(pred, target, window_size=11):
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    kernel = torch.ones((1, 1, window_size, window_size), device=pred.device) / (window_size ** 2)
    kernel = kernel.repeat(pred.shape[1], 1, 1, 1)

    mu_x = F.conv2d(pred, kernel, padding=window_size // 2, groups=pred.shape[1])
    mu_y = F.conv2d(target, kernel, padding=window_size // 2, groups=pred.shape[1])

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=pred.shape[1]) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, kernel, padding=window_size // 2, groups=pred.shape[1]) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=pred.shape[1]) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
    return ssim_map.mean().item()


def compute_psnr(mse):
    if mse == 0:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def test_cyclegan(model_path, data_path, out_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing CycleGAN generator on {device}...")

    checkpoint_path = model_path
    test_dir = data_path
    output_dir = out_path
    os.makedirs(output_dir, exist_ok=True)

    # Instantiate generator G_I2E: takes 2-channel input (I0, I1) -> 2-channel events
    generator = ResNetGenerator(in_channels=2, out_channels=2).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(ckpt['G_I2E_state_dict'])
    generator.eval()

    test_dataset = DAVIS240CTestDataset(base_dir=test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    total_mae = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    num_samples = len(test_loader)
    num_plots = 15

    print(f"Evaluating {num_samples} frame pairs from {test_dir}...")

    with torch.no_grad():
        for i, (img0, img1, real_events) in enumerate(test_loader):
            img0 = img0.to(device)
            img1 = img1.to(device)
            real_events = real_events.to(device)

            # Input frame pair: (B, 2, H, W)
            frame_pair = torch.cat([img0, img1], dim=1)
            fake_events = generator(frame_pair)

            mae = F.l1_loss(fake_events, real_events).item()
            mse = F.mse_loss(fake_events, real_events).item()
            ssim_val = compute_ssim(fake_events, real_events)

            total_mae += mae
            total_mse += mse
            total_ssim += ssim_val

            # Plot visual comparisons for the first 15 frame pairs
            if i < num_plots:
                frame0_np = img0[0, 0].cpu().numpy()
                frame1_np = img1[0, 0].cpu().numpy()
                
                # Direct temporal frame difference (|I_{t+1} - I_t|)
                temporal_diff = np.abs(frame1_np - frame0_np)

                # Ground truth events RGB visualization: Pos = Red, Neg = Blue
                real_rgb = np.zeros((180, 240, 3), dtype=np.float32)
                real_rgb[:, :, 0] = real_events[0, 0].cpu().numpy()  # Pos (R)
                real_rgb[:, :, 2] = real_events[0, 1].cpu().numpy()  # Neg (B)

                # CycleGAN predicted events RGB visualization
                fake_rgb = np.zeros((180, 240, 3), dtype=np.float32)
                fake_rgb[:, :, 0] = fake_events[0, 0].cpu().numpy()  # Pos (R)
                fake_rgb[:, :, 2] = fake_events[0, 1].cpu().numpy()  # Neg (B)

                fig, axes = plt.subplots(1, 4, figsize=(18, 4))

                axes[0].imshow(frame0_np, cmap='gray')
                axes[0].set_title(f"Input Frame $I_t$ ({i:03d})")
                axes[0].axis('off')

                axes[1].imshow(temporal_diff, cmap='inferno')
                axes[1].set_title(r"Absolute Frame Diff $|I_{t+1} - I_t|$")
                axes[1].axis('off')

                axes[2].imshow(np.clip(real_rgb, 0.0, 1.0))
                axes[2].set_title("Real Events (Red:+, Blue:-)")
                axes[2].axis('off')

                axes[3].imshow(np.clip(fake_rgb, 0.0, 1.0))
                axes[3].set_title(f"CycleGAN Pred (SSIM: {ssim_val:.3f})")
                axes[3].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"cyclegan_frame_{i:03d}.png"), dpi=150)
                plt.close(fig)

    avg_mae = total_mae / num_samples
    avg_mse = total_mse / num_samples
    avg_rmse = np.sqrt(avg_mse)
    avg_ssim = total_ssim / num_samples
    avg_psnr = compute_psnr(avg_mse)

    print("\n================== CycleGAN Evaluation Summary (hdr_boxes) ==================")
    print(f"Total Samples Evaluated:            {num_samples}")
    print(f"Mean Absolute Error (MAE):          {avg_mae:.5f}")
    print(f"Root Mean Squared Error (RMSE):     {avg_rmse:.5f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR):  {avg_psnr:.2f} dB")
    print(f"Mean SSIM:                          {avg_ssim:.5f}")
    print(f"Visual outputs saved to:            {output_dir}")
    print("================================================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)

    args = parser.parse_args()

    test_cyclegan(args.model_path,
        args.data_path,
        args.out_path)