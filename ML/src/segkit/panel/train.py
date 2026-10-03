"""Train PanelNet on segkit-panel-label sessions, then export the best checkpoint to ExecuTorch.

    segkit-panel-train data/panel/<session> [...] --negatives data/panel/negatives.txt --out runs/panel_v1

Losses: cell area (BCE + dice), gridlines (BCE + dice), within-cell phase (MSE on the sin/cos pairs, inside the
cell area only) and panel-present (BCE). Validation reports cell-area IoU, the median phase error in cells
(how far off the predicted position inside a cell is) and presence accuracy on held-out time blocks.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from segkit.datasets.hands import IMAGENET_MEAN, IMAGENET_STD, read_list
from segkit.datasets.panels import PanelCrops
from segkit.losses import bce_dice
from segkit.models.panelnet import PanelNet, PanelProbabilityHead

LINE_WEIGHT = 0.5
PHASE_WEIGHT = 1.0
PRESENCE_WEIGHT = 0.5
N_VIS = 6


def loss_fn(dense: torch.Tensor, present_logit: torch.Tensor, t: torch.Tensor, present: torch.Tensor):
    dense = dense.float()
    mask = t[:, 0:1]
    l_mask = bce_dice(dense[:, 0:1], mask)
    l_lines = bce_dice(dense[:, 1:2], t[:, 1:2])
    w = mask.expand(-1, 4, -1, -1)
    l_phase = ((dense[:, 2:6] - t[:, 2:6]) ** 2 * w).sum() / w.sum().clamp(min=1)
    l_pres = F.binary_cross_entropy_with_logits(present_logit.float(), present)
    return l_mask + LINE_WEIGHT * l_lines + PHASE_WEIGHT * l_phase + PRESENCE_WEIGHT * l_pres


def phase_error_cells(dense: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Per-pixel distance between predicted and true within-cell position, in cells (u and v combined)."""
    du = torch.atan2(dense[:, 2], dense[:, 3]) - torch.atan2(t[:, 2], t[:, 3])
    dv = torch.atan2(dense[:, 4], dense[:, 5]) - torch.atan2(t[:, 4], t[:, 5])
    wrap = lambda a: torch.remainder(a + math.pi, 2 * math.pi) - math.pi
    return torch.hypot(wrap(du) / (2 * math.pi), wrap(dv) / math.pi)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, list]:
    model.eval()
    losses, ious, perr, pres_ok, vis = [], [], [], [], []
    for x, t, present in loader:
        x, t, present = x.to(device), t.to(device), present.to(device)
        dense, pl = model(x)
        losses.append(loss_fn(dense, pl, t, present).item())
        pred = dense[:, 0] > 0
        gt = t[:, 0] > 0.5
        pres_ok += ((pl[:, 0] > 0) == (present[:, 0] > 0.5)).tolist()
        err = phase_error_cells(dense, t)
        for b in range(len(x)):
            if gt[b].sum() > 100:
                ious.append(((pred[b] & gt[b]).sum() / (pred[b] | gt[b]).sum()).item())
                perr.append(err[b][gt[b]].median().item())
        if len(vis) < N_VIS:
            vis += [(x[b].cpu().numpy(), t[b].cpu().numpy(), dense[b].cpu().numpy())
                    for b in range(len(x)) if gt[b].sum() > 100][:N_VIS - len(vis)]
    return {"val_loss": float(np.mean(losses)), "iou": float(np.mean(ious)),
            "phase_err_cells": float(np.median(perr)), "present_acc": float(np.mean(pres_ok))}, vis


def phase_image(su, cu, sv, cv, mask) -> np.ndarray:
    hue_u = (np.arctan2(su, cu) / (2 * math.pi) % 1)
    hue_v = (np.arctan2(sv, cv) / (2 * math.pi) % 1)
    img = np.stack([hue_u * 179, np.full_like(hue_u, 255), 255 * (0.5 + 0.5 * hue_v)], -1).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR) * mask[..., None].astype(np.uint8)


def save_vis(vis: list, path: Path) -> None:
    rows = []
    for x, t, d in vis:
        rgb = ((x.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ov = bgr.copy()
        ov[d[1] > 0] = (0, 0, 255)
        contours, _ = cv2.findContours((d[0] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, contours, -1, (0, 255, 0), 2)
        rows.append(np.hstack([bgr, ov, phase_image(*t[2:6], t[0] > 0.5), phase_image(*d[2:6], d[0] > 0)]))
    cv2.imwrite(str(path), np.vstack(rows))


def main() -> None:
    p = argparse.ArgumentParser(prog="segkit-panel-train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions", type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=Path("runs/panel"))
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--export-size", type=int, default=384, help="input size of the exported .pte (phone crop)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--negatives", type=Path, help="list of panel-free image paths")
    p.add_argument("--negative-share", type=float, default=0.2)
    p.add_argument("--steps-per-epoch", type=int, default=0, help="default: one pass over the panel frames")
    p.add_argument("--init", type=Path, help="start from these weights (fine-tune) instead of ImageNet")
    p.add_argument("--encoder", default="mobilenetv3_small_100",
                   help="timm encoder; the phone's fast path uses mobilenetv3_small_075 at --size 192")
    p.add_argument("--decoder", default="96,64,32", help="decoder channels at 1/16, 1/8, 1/4")
    p.add_argument("--name", default="panelseg", help="exported file name (<name>.pte)")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--grazing", action="store_true", help="add physical plane tilts up to 72 degrees")
    p.add_argument("--rot90", action="store_true", help="also train on quarter-turned views (phone held sideways)")
    p.add_argument("--val-sessions", type=Path, nargs="+", help="checkpoint selection sessions; other validation blocks remain untouched")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cv2.setNumThreads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    negatives = read_list(args.negatives)
    n_val_neg = len(negatives) // 10
    rng = np.random.default_rng(0)
    rng.shuffle(negatives)
    train_set = PanelCrops(args.sessions, "train", args.size, True, negatives[n_val_neg:], grazing=args.grazing,
                           rot90=args.rot90)
    val_set = PanelCrops(args.val_sessions or args.sessions, "val", args.size, False, negatives[:n_val_neg])
    print(f"device {device}  train {train_set.n_panel} panel + {len(train_set) - train_set.n_panel} negative  "
          f"val {val_set.n_panel} + {len(val_set) - val_set.n_panel}  size {args.size}")
    n_neg = len(train_set) - train_set.n_panel
    weights = [1.0] * train_set.n_panel
    if n_neg:
        weights += [train_set.n_panel * args.negative_share / (1 - args.negative_share) / n_neg] * n_neg
    steps = args.steps_per_epoch or train_set.n_panel // args.batch
    sampler = WeightedRandomSampler(weights, num_samples=steps * args.batch, replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                              drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=args.workers)

    decoder = tuple(int(c) for c in args.decoder.split(","))
    model = PanelNet(args.encoder, pretrained=args.init is None, decoder_channels=decoder).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
    print(f"PanelNet {args.encoder} {decoder}: {sum(q.numel() for q in model.parameters()) / 1e6:.2f}M params")
    (args.out / "arch.txt").write_text(f"{args.encoder} {args.decoder} {args.export_size}\n")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
    best = float("inf")
    with (args.out / "metrics.csv").open("w", newline="") as f:
        log = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "iou", "phase_err_cells",
                                            "present_acc", "sec"])
        log.writeheader()
        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            model.train()
            tl = []
            for x, t, present in train_loader:
                x, t, present = x.to(device), t.to(device), present.to(device)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    dense, pl = model(x)
                loss = loss_fn(dense, pl, t, present)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tl.append(loss.item())
            m, vis = evaluate(model, val_loader, device)
            row = {"epoch": epoch, "train_loss": float(np.mean(tl)), **m, "sec": time.perf_counter() - t0}
            log.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})
            f.flush()
            # Best = sharp position inside the cell on a correctly found panel.
            score = m["phase_err_cells"] + (1 - m["iou"]) + (1 - m["present_acc"])
            improved = score < best
            if improved:
                best = score
                torch.save(model.state_dict(), args.out / "best.pt")
                save_vis(vis, args.out / "val_best.jpg")
            torch.save(model.state_dict(), args.out / "last.pt")
            print(f"epoch {epoch:3d}  train {row['train_loss']:.4f}  val {m['val_loss']:.4f}  IoU {m['iou']:.4f}  "
                  f"phase err {m['phase_err_cells']:.4f} cells  present {m['present_acc']:.3f}  "
                  f"({row['sec']:.0f}s){'  *best' if improved else ''}", flush=True)

    if not args.no_export:
        from segkit.export import PteRunner, export_pte
        model.load_state_dict(torch.load(args.out / "best.pt", map_location="cpu"))
        deploy = PanelProbabilityHead(model.cpu()).eval()
        x = torch.randn(1, 3, args.export_size, args.export_size)
        pte = export_pte(deploy, x, args.out / f"{args.name}.pte")
        with torch.no_grad():
            eager = deploy(x)
        out = PteRunner(pte).method.execute([x])
        diff = max((a - b).abs().max().item() for a, b in zip(eager, out))
        print(f"exported {pte} ({pte.stat().st_size / 1e6:.2f} MB), max |eager - pte| = {diff:.2e}")


if __name__ == "__main__":
    main()
