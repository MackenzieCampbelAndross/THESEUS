"""Beacon detector: background suppression -> adaptive threshold -> blobs -> centroid + confidence."""
from dataclasses import dataclass
import math
import cv2
import numpy as np
from .config import DetectorCfg


@dataclass
class Detection:
    u: float
    v: float
    conf: float
    snr: float
    area: int


class BeaconDetector:
    def __init__(self, cfg: DetectorCfg):
        self.c = cfg
        self.last_sigma = 0.0
        self.fpn = None

    def calibrate(self, dark_mean):
        """Fixed-pattern (hot pixel) map from averaged dark frames; subtracted from every frame."""
        self.fpn = dark_mean - float(np.median(dark_mean))

    def detect(self, gray):
        c = self.c
        img = gray.astype(np.float32)
        if self.fpn is not None:
            img = img - self.fpn
        sm = cv2.GaussianBlur(img, (0, 0), c.blur_sigma)
        bg = cv2.boxFilter(sm, -1, (c.bg_kernel, c.bg_kernel), normalize=True)
        res = sm - bg
        sub = res[::4, ::4]
        med = float(np.median(sub))
        sigma = max(1.4826 * float(np.median(np.abs(sub - med))), 0.3)   # robust noise level (MAD)
        self.last_sigma = sigma
        thr = med + c.k_sigma * sigma
        mask = (res > thr).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cands = []
        raw = img - bg
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < c.min_area or a > c.max_area:
                continue
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > c.max_aspect:
                continue
            x0, y0 = max(x - 2, 0), max(y - 2, 0)
            x1, y1 = min(x + w + 2, img.shape[1]), min(y + h + 2, img.shape[0])
            wgt = np.clip(raw[y0:y1, x0:x1] - thr * 0.5, 0, None)
            tot = float(wgt.sum())
            if tot <= 0:
                continue
            ys, xs = np.mgrid[y0:y1, x0:x1]
            u = float((wgt * xs).sum() / tot)
            v = float((wgt * ys).sum() / tot)
            peak = float(res[lab == i].max())
            snr = (peak - med) / sigma
            conf = 1.0 / (1.0 + math.exp(-(snr - c.snr0) / c.snr_scale))
            cands.append(Detection(u, v, conf, snr, int(a)))
        cands.sort(key=lambda d: -d.conf)
        return cands[: c.max_candidates]
