"""Paired model evaluation on held-out blocks only, including independently reset tracker runs.

python -m segkit.panel.compare --baseline runs/panel_s2/best.pt --candidate runs/panel_s3/best.pt \
    --original data/panel/20261002_152323 --new data/panel/20261003_163008 \
    --negatives data/panel/negatives.remote.txt --out runs/panel_s3/comparison
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from segkit.datasets.hands import read_list, normalize
from segkit.datasets.panels import PanelCrops, is_val, load_session, session_is_crops, targets
from segkit.models.panelnet import PanelNet, PanelProbabilityHead
from segkit.panel.train import evaluate, phase_error_cells
from segkit.panel.track import PanelTracker, centre_crop, cell_at, draw
from segkit.panel import geometry as G


def gate(old, new):
    """Predeclared material regression limits; all must pass before replacing the asset."""
    original, oblique = [], []
    a, b = old['original'], new['original']
    original += [b['iou'] >= a['iou'] - .005, b['phase_err_cells'] <= a['phase_err_cells'] + .01,
                 b['present_acc'] >= a['present_acc'] - .005]
    a, b = old['new'], new['new']
    oblique += [b['iou'] > a['iou'] + .02, b['phase_err_cells'] <= a['phase_err_cells'],
                b['present_acc'] >= a['present_acc']]
    for split in ['original_tracker', 'new_tracker']:
        a, b = old[split], new[split]
        original += [b['correct_centre'] >= a['correct_centre'], b['good_grid_frames'] >= a['good_grid_frames'],
                     b['wrong_centre'] <= a['wrong_centre']]
    return {'original_and_tracker_checks': original, 'oblique_checks': oblique,
            'replace_asset': bool(all(original + oblique))}


@torch.no_grad()
def tracker_eval(model, session, out):
    paths, Hs, valid, spec = load_session(session)
    crops = session_is_crops(session)
    head = PanelProbabilityHead(model).eval()
    tracker = None
    rows = []
    images = []
    previous = -2
    for i, path in enumerate(paths):
        if not is_val(i):
            continue
        # Never warm up on training frames; every held-out block starts lost.
        if i != previous + 1:
            tracker = PanelTracker(spec, 384, klt=False, block_match=True)
            cv2.setRNGSeed(0)
        previous = i
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        if crops:
            crop = cv2.resize(rgb, (192, 192))
            A = np.diag([192 / rgb.shape[1], 192 / rgb.shape[0], 1.])
        else:
            crop, A = centre_crop(rgb, 192)
        d, p = head(normalize(crop)[None].to(next(model.parameters()).device))
        dense = d[0].cpu().numpy()
        large = cv2.resize(crop, (384, 384))
        dense_large = np.stack([cv2.resize(channel, (384, 384), interpolation=cv2.INTER_NEAREST) for channel in dense])
        result = tracker.step(large, dense_large, float(p[0, 0]))
        row = {'frame': i, 'valid': bool(valid[i]), 'state': result.state, 'centre': result.centre_cell,
               'label_centre': None, 'grid_error_cells': None, 'good_grid': False}
        if valid[i]:
            truth = np.diag([2., 2., 1.]) @ A @ Hs[i]
            row['label_centre'] = cell_at(truth, spec, 192, 192)
            if result.H is not None:
                t = targets(truth, 384, spec)[0]
                y, x = np.nonzero(t[::8, ::8] > .5)
                xy = np.c_[x * 8 + .5, y * 8 + .5]
                if len(xy):
                    gt = G.unproject(truth, xy)
                    pred = G.unproject(result.H, xy)
                    error = np.median(np.linalg.norm(pred - gt, axis=1))
                    row['grid_error_cells'] = float(error) if np.isfinite(error) else None
                    row['good_grid'] = bool(np.isfinite(error) and error < .25)
            if len(images) < 24:
                images.append(draw(cv2.cvtColor(large, cv2.COLOR_RGB2BGR), result, spec, dense_large, row['label_centre']))
        rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(rows, indent=2))
    if images:
        while len(images) % 4:
            images.append(np.zeros_like(images[0]))
        cv2.imwrite(str(out.with_suffix('.jpg')), np.vstack([np.hstack(images[i:i+4]) for i in range(0,len(images),4)]))
    scored = [r for r in rows if r['valid']]
    centre = [r for r in scored if r['label_centre'] is not None]
    said = [r for r in scored if r['centre'] is not None]
    return {'frames':len(rows), 'labelled':len(scored), 'centre_on_panel':len(centre),
            'claimed_centre':len(said), 'correct_centre':sum(r['centre']==r['label_centre'] for r in said),
            'wrong_centre':sum(r['centre']!=r['label_centre'] for r in said),
            'good_grid_frames':sum(r['good_grid'] for r in scored), 'states':dict(Counter(r['state'] for r in rows))}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','candidate','original','new','negatives','out']:
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    negatives=read_list(args.negatives);np.random.default_rng(0).shuffle(negatives)
    sets={'original':PanelCrops([args.original],'val',192,False,negatives[:len(negatives)//10]),
          'new':PanelCrops([args.new],'val',192,False)}
    report={'protocol':'checkpoint chosen only using original validation; tracker resets at each held-out block; no training-frame warmup',
            'thresholds':{'original_iou_drop':.005,'original_phase_increase':.01,'original_presence_drop':.005,
                          'new_iou_gain':.02,'new_phase_and_presence':'no regression','tracker':'correct cells/good grids cannot drop; wrong cells cannot increase'},
            'new_heldout_files':[p.name for p,H in sets['new'].items]}
    args.out.mkdir(parents=True,exist_ok=True)
    for name,path in [('baseline',args.baseline),('candidate',args.candidate)]:
        model=PanelNet(pretrained=False).to(device)
        model.load_state_dict(torch.load(path,map_location=device));model.eval()
        metrics={}
        for split,dataset in sets.items():
            metrics[split]=evaluate(model,DataLoader(dataset,batch_size=16,num_workers=2),device)[0]
        for split,session in [('original',args.original),('new',args.new)]:
            metrics[split+'_tracker']=tracker_eval(model,session,args.out/f'{name}_{split}_tracker')
        report[name]=metrics
        print(name,json.dumps(metrics),flush=True)
    report['gate']=gate(report['baseline'],report['candidate'])
    (args.out/'comparison.json').write_text(json.dumps(report,indent=2))
    print('gate',report['gate'],flush=True)

if __name__=='__main__':main()
