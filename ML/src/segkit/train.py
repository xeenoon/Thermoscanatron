"""Train HandSegNet on a segkit-label dataset, then export the best checkpoint to ExecuTorch.

segkit-train data/hands_all --out runs/handseg_v1 --epochs 40

The model outputs a mask and a hand-present score; no-hand crops train both towards "nothing here".
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from segkit.datasets.hands import (IMAGENET_MEAN, IMAGENET_STD, HandCrops, NegativeImages, load_split, read_list,
                                   session_of)
from segkit.eval.metrics import score
from segkit.losses import bce_dice
from segkit.models.handseg import HandSegNet, ProbabilityHead

N_VIS = 8
PRESENCE_LOSS_WEIGHT = 0.5


def loss_fn(mask_logits: torch.Tensor, present_logit: torch.Tensor, mask: torch.Tensor,
            present: torch.Tensor) -> torch.Tensor:
    return (bce_dice(mask_logits.float(), mask)
            + PRESENCE_LOSS_WEIGHT * F.binary_cross_entropy_with_logits(present_logit.float(), present))


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[dict, list]:
    """Boundary metrics on crops with a hand; presence accuracy on all; false positives on no-hand crops."""
    model.eval()
    ious, p95s, f2s, losses, vis = [], [], [], [], []
    present_ok, neg_false_pos = [], []
    for x, m, present in loader:
        x, m, present = (t.to(device, non_blocking=True) for t in (x, m, present))
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            mask_logits, present_logit = model(x)
        losses.append(loss_fn(mask_logits, present_logit, m, present).item())
        pred = (mask_logits.float() > 0).cpu().numpy()[:, 0]
        pred_present = (present_logit.float() > 0).cpu().numpy()[:, 0]
        gt = m.cpu().numpy()[:, 0] > 0.5
        gt_present = present.cpu().numpy()[:, 0] > 0.5
        present_ok += list(pred_present == gt_present)
        for p, g, gp, pp in zip(pred, gt, gt_present, pred_present):
            if gp:
                s = score(p, g)
                ious.append(s.iou)
                p95s.append(s.p95_px)
                f2s.append(s.f_at[2])
            else:
                # A no-hand crop is a false positive if either head claims a hand.
                neg_false_pos.append(bool(pp) or p.mean() >= 0.005)
        if len(vis) < N_VIS:
            vis += [v for v in zip(x.cpu().numpy(), gt, pred, gt_present) if v[3]][:N_VIS - len(vis)]
    return {"val_loss": float(np.mean(losses)), "iou": float(np.mean(ious)),
            "f@2px": float(np.mean(f2s)), "p95_px": float(np.median(p95s)),
            "present_acc": float(np.mean(present_ok)),
            "neg_fp_rate": float(np.mean(neg_false_pos)) if neg_false_pos else float("nan")}, vis


def save_vis(vis: list, path: Path) -> None:
    tiles = []
    for x, gt, pred, _ in vis:
        rgb = ((x.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for mask, colour in ((gt, (0, 255, 0)), (pred, (0, 0, 255))):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(bgr, contours, -1, colour, 1)
        tiles.append(bgr)
    cv2.imwrite(str(path), np.hstack(tiles))


def main() -> None:
    p = argparse.ArgumentParser(prog="segkit-train")
    p.add_argument("dataset", type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/handseg"))
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=0, help="use only this many train/val samples (smoke test)")
    p.add_argument("--negatives", type=Path, help="list of hand-free image paths (segkit-negatives) for training")
    p.add_argument("--val-negatives", type=Path, help="held-out list of hand-free image paths for validation")
    p.add_argument("--negative-share", type=float, default=0.25,
                   help="share of each training epoch drawn from --negatives")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--init", type=Path, help="start from these weights (fine-tune), e.g. the 384 px model's best.pt "
                                             "for the phone's small fast-path model at --size 192")
    p.add_argument("--name", default="handseg", help="exported file name (<name>.pte)")
    p.add_argument("--no-export", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    train_stems, val_stems = load_split(args.dataset)
    if args.limit:
        train_stems, val_stems = train_stems[:args.limit], val_stems[:max(1, args.limit // 4)]
    print(f"device {device}  train {len(train_stems)}  val {len(val_stems)}  size {args.size}")
    def n_no_hand(stems: list[str]) -> int:
        return sum(not cv2.imread(str(args.dataset / "masks" / f"{s}.png"), cv2.IMREAD_GRAYSCALE).any() for s in stems)
    print(f"no-hand frames: train {n_no_hand(train_stems)}  val {n_no_hand(val_stems)}")

    # Every recording session gets the same weight per epoch, however many frames it has.
    sessions = [session_of(s) for s in train_stems]
    counts = {k: sessions.count(k) for k in set(sessions)}
    print("train frames per session: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    negatives = read_list(args.negatives)
    weights = [1.0 / counts[k] for k in sessions]
    train_set = HandCrops(args.dataset, train_stems, args.size, train=True, negatives=negatives)
    if negatives:
        # Hand frames keep their total weight; negatives get negative_share of the draws, spread evenly.
        own = sum(weights)
        neg_w = own * args.negative_share / (1 - args.negative_share) / len(negatives)
        weights += [neg_w] * len(negatives)
        train_set = ConcatDataset([train_set, NegativeImages(negatives, args.size, train=True)])
        print(f"negatives: {len(negatives)} images, {args.negative_share:.0%} of draws, also used as paste backgrounds")
    sampler = WeightedRandomSampler(weights, num_samples=len(train_stems), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch,
                              sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    # Validation: one centred crop per frame, plus a fixed background crop from every 2nd hand frame so the
    # "no hand" false-positive rate is measured on real cluttered backgrounds.
    val_parts = [HandCrops(args.dataset, val_stems, args.size, train=False),
                 HandCrops(args.dataset, val_stems[::2], args.size, train=False, background_only=True)]
    val_negatives = read_list(args.val_negatives)
    if val_negatives:
        val_parts.append(NegativeImages(val_negatives, args.size, train=False))
    val_set = ConcatDataset(val_parts)
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=args.workers, pin_memory=True)

    model = HandSegNet(pretrained=not args.no_pretrained and args.init is None).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
    print(f"HandSegNet {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_loader),
                                                pct_start=0.1)
    writer = SummaryWriter(args.out / "tb")
    log_path = args.out / "metrics.csv"
    best_f2 = -1.0

    with log_path.open("w", newline="") as log_file:
        log = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "val_loss", "iou", "f@2px", "p95_px",
                                                   "present_acc", "neg_fp_rate", "sec"])
        log.writeheader()
        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            model.train()
            train_losses = []
            for x, m, present in train_loader:
                x, m, present = (t.to(device, non_blocking=True) for t in (x, m, present))
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    mask_logits, present_logit = model(x)
                loss = loss_fn(mask_logits, present_logit, m, present)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                train_losses.append(loss.item())

            metrics, vis = evaluate(model, val_loader, device)
            row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **metrics,
                   "sec": time.perf_counter() - t0}
            log.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})
            log_file.flush()
            for k, v in row.items():
                if k != "epoch":
                    writer.add_scalar(k, v, epoch)
            # Best = boundary quality on hands x getting "is there a hand" right.
            selection = metrics["f@2px"] * metrics["present_acc"]
            improved = selection > best_f2
            if improved:
                best_f2 = selection
                torch.save(model.state_dict(), args.out / "best.pt")
                save_vis(vis, args.out / "val_best.jpg")
            torch.save(model.state_dict(), args.out / "last.pt")
            print(f"epoch {epoch:3d}  train {row['train_loss']:.4f}  val {metrics['val_loss']:.4f}  "
                  f"IoU {metrics['iou']:.4f}  F@2px {metrics['f@2px']:.4f}  p95 {metrics['p95_px']:.2f}px  "
                  f"present {metrics['present_acc']:.3f}  negFP {metrics['neg_fp_rate']:.3f}  "
                  f"({row['sec']:.0f}s){'  *best' if improved else ''}", flush=True)

    if not args.no_export:
        from segkit.export import PteRunner, export_pte
        model.load_state_dict(torch.load(args.out / "best.pt", map_location="cpu"))
        deploy = ProbabilityHead(model.cpu()).eval()
        x = torch.randn(1, 3, args.size, args.size)
        pte = export_pte(deploy, x, args.out / f"{args.name}.pte")
        with torch.no_grad():
            eager = deploy(x)
        out = PteRunner(pte).method.execute([x])
        diff = max((a - b).abs().max().item() for a, b in zip(eager, out))
        print(f"exported {pte} ({pte.stat().st_size / 1e6:.2f} MB), max |eager - pte| = {diff:.2e}")


if __name__ == "__main__":
    main()
