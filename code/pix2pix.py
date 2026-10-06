import argparse
import os
import bisect
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


class FixedSobelEdgeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], 
                                [-2., 0., 2.], 
                                [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], 
                                [ 0.,  0.,  0.], 
                                [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
        self.register_buffer('kx', sobel_x)
        self.register_buffer('ky', sobel_y)

    def forward(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        return torch.cat([mag, gx, gy], dim=1)


class EventSynthesisGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_decoder = FixedSobelEdgeDecoder()
        in_c = 7

        def down_block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, kernel_size=4, stride=2, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_f, out_f):
            return nn.Sequential(
                nn.ConvTranspose2d(in_f, out_f, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_f),
                nn.ReLU(inplace=True)
            )

        self.down1 = down_block(in_c, 64, normalize=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)

        self.up1 = up_block(512, 256)
        self.up2 = up_block(512, 128)
        self.up3 = up_block(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img0, img1):
        with torch.no_grad():
            e0 = self.edge_decoder(img0)
            e1 = self.edge_decoder(img1)
            delta_mag = e1[:, 0:1] - e0[:, 0:1]
            edge_features = torch.cat([e0, e1, delta_mag], dim=1)

        d1 = self.down1(edge_features)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4)
        if u1.shape[2:] != d3.shape[2:]:
            u1 = F.interpolate(u1, size=d3.shape[2:], mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([u1, d3], dim=1))

        if u2.shape[2:] != d2.shape[2:]:
            u2 = F.interpolate(u2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        u3 = self.up3(torch.cat([u2, d2], dim=1))

        if u3.shape[2:] != d1.shape[2:]:
            u3 = F.interpolate(u3, size=d1.shape[2:], mode='bilinear', align_corners=False)
        out = self.final(torch.cat([u3, d1], dim=1))

        if out.shape[2:] != img0.shape[2:]:
            out = F.interpolate(out, size=img0.shape[2:], mode='bilinear', align_corners=False)
        return edge_features, out


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

        img0 = Image.open(path0).convert('L')
        img1 = Image.open(path1).convert('L')

        img0_tensor = TF.to_tensor(img0)
        img1_tensor = TF.to_tensor(img1)

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

        event_frame = np.log1p(event_frame)
        if event_frame.max() > 0:
            event_frame = event_frame / event_frame.max()

        return img0_tensor, img1_tensor, torch.from_numpy(event_frame)


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



def test_code(model_path, data_path, out_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    test_dir = data_path
    checkpoint_path = model_path
    output_dir = out_path
    os.makedirs(output_dir, exist_ok=True)

    generator = EventSynthesisGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    test_dataset = DAVIS240CTestDataset(base_dir=test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    total_mae, total_mse, total_ssim = 0.0, 0.0, 0.0
    num_samples = len(test_loader)

    print(f"Evaluating {num_samples} frame pairs from boxes_rotation...")

    with torch.no_grad():
        for i, (img0, img1, real_events) in enumerate(test_loader):
            img0 = img0.to(device)
            img1 = img1.to(device)
            real_events = real_events.to(device)

            edge_cond, fake_events = generator(img0, img1)

            mae = F.l1_loss(fake_events, real_events).item()
            mse = F.mse_loss(fake_events, real_events).item()
            ssim_val = compute_ssim(fake_events, real_events)

            total_mae += mae
            total_mse += mse
            total_ssim += ssim_val

            # Save qualitative visual comparison every 50 frames
            if i % 50 == 0:
                frame_i = img0[0, 0].cpu().numpy()
                edge_mag = edge_cond[0, 0].cpu().numpy()

                # Merge pos (Red) and neg (Blue) into an RGB event frame representation
                real_ev_rgb = np.zeros((180, 240, 3))
                real_ev_rgb[:, :, 0] = real_events[0, 0].cpu().numpy()  # Pos = Red
                real_ev_rgb[:, :, 2] = real_events[0, 1].cpu().numpy()  # Neg = Blue

                fake_ev_rgb = np.zeros((180, 240, 3))
                fake_ev_rgb[:, :, 0] = fake_events[0, 0].cpu().numpy()
                fake_ev_rgb[:, :, 2] = fake_events[0, 1].cpu().numpy()

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(frame_i, cmap='gray')
                axes[0].set_title(f"Input Frame {i}")
                axes[0].axis('off')

                axes[1].imshow(edge_mag, cmap='inferno')
                axes[1].set_title("Sobel Edge Filter")
                axes[1].axis('off')

                axes[2].imshow(real_ev_rgb)
                axes[2].set_title("Ground Truth Events (R/B)")
                axes[2].axis('off')

                axes[3].imshow(fake_ev_rgb)
                axes[3].set_title("Predicted Events (R/B)")
                axes[3].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"comparison_sample_{i:04d}.png"), dpi=150)
                plt.close(fig)

    avg_mae = total_mae / num_samples
    avg_rmse = np.sqrt(total_mse / num_samples)
    avg_ssim = total_ssim / num_samples

    print("\n--- Test Evaluation Summary (boxes_rotation) ---")
    print(f"Mean Absolute Error (MAE):     {avg_mae:.5f}")
    print(f"Root Mean Squared Error (RMSE): {avg_rmse:.5f}")
    print(f"Mean SSIM:                     {avg_ssim:.5f}")
    print(f"Visual plots saved to:          {output_dir}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    
    args = parser.parse_args()

    test_code(args.model_path, args.data_path, args.out_path)