import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as f


def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = valid_dataloader(args.data_dir, args.data, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():
        print('Start Evaluation')

        for idx, data in enumerate(dataset):
            input_img, label_img = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            # =========================
            # 非 OHAZE 数据集（保持原逻辑）
            # =========================
            if args.data != 'OHAZE':
                factor = 32
                h, w = input_img.shape[2], input_img.shape[3]
                H = ((h + factor - 1) // factor) * factor
                W = ((w + factor - 1) // factor) * factor
                padh = H - h
                padw = W - w

                if padh > 0 or padw > 0:
                    input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

                pred = model(input_img)[2]
                pred = pred[:, :, :h, :w]

            # =========================
            # OHAZE 数据集（固定裁剪）
            # =========================

            else:
                # 固定裁剪
                input_img = input_img[:, :, 100:2048, 100:1950]
                label_img = label_img[:, :, 100:2048, 100:1950]

                # ===== 关键：补齐到32倍数 =====
                factor = 32
                h, w = input_img.shape[2], input_img.shape[3]

                H = ((h + factor - 1) // factor) * factor
                W = ((w + factor - 1) // factor) * factor

                padh = H - h
                padw = W - w

                if padh > 0 or padw > 0:
                    input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

                pred = model(input_img)[2]

                # 裁剪回原尺寸
                pred = pred[:, :, :h, :w]

            # =========================
            # PSNR 计算
            # =========================
            pred_clip = torch.clamp(pred, 0, 1)

            psnr = 10 * torch.log10(
                1 / f.mse_loss(pred_clip, label_img)
            )

            psnr_adder(psnr.item())
            print(f'\r{idx:03d}', end=' ')

            del input_img, label_img, pred
            torch.cuda.empty_cache()

    print('\n')
    model.train()
    return psnr_adder.average()