import math

import numpy as np

from segkit.datasets.panels import targets
from segkit.panel import geometry as G
from segkit.panel import thermal as T
from segkit.panel.spec import PanelSpec
from segkit.panel.track import unwrap

SPEC = PanelSpec()


def front_on_homography(size=384):
    """Panel filling most of a size x size image, upright, front on."""
    cell = size / 12
    return np.array([[cell, 0, size / 2 - 2 * cell], [0, cell * 0.69, size / 2 - 4.5 * cell * 0.69], [0, 0, 1]])


def test_targets_phase_matches_geometry():
    H = front_on_homography()
    t = targets(H, 384, SPEC)
    uv = G.unproject(H, np.array([[192.5, 192.5]]))[0]
    assert t[0, 192, 192] == 1
    assert math.isclose(math.atan2(t[2, 192, 192], t[3, 192, 192]) % (2 * math.pi),
                        (2 * math.pi * uv[0]) % (2 * math.pi), abs_tol=1e-3)
    assert t[0, 0, 0] == 0


def test_unwrap_recovers_coordinates_up_to_offset():
    H = front_on_homography(96)
    t = targets(H, 96, SPEC)
    mask = t[0] > 0.5
    u, v = unwrap(mask, np.arctan2(t[2], t[3]) / (2 * math.pi), np.arctan2(t[4], t[5]) / math.pi)
    ys, xs = np.nonzero(mask)
    true = G.unproject(H, np.c_[xs + 0.5, ys + 0.5])
    du = u[ys, xs] - true[:, 0]
    dv = v[ys, xs] - true[:, 1]
    assert np.ptp(np.round(du)) == 0 and np.abs(du - np.round(du)).max() < 1e-3
    assert np.ptp(np.round(dv)) == 0 and round(dv[0]) % 2 == 0


def test_hotspot_flagged():
    """Thermal camera at the phone lens looking straight at a panel 1 m away: one cell 10 C hotter."""
    calib = {"R": np.eye(3), "c_m": np.zeros(3), "thermal_cx": 15.5, "thermal_cy": 11.5,
             "thermal_fx": 33.3, "thermal_fy": 39.2, "mirror": False}
    rays = T.footprint_rays(calib)
    # Panel centred on the axis at 1 m: origin at (-2 cells, -4.5 rows), axes along x and y.
    t = np.array([-2 * SPEC.cell_w_m, -4.5 * SPEC.cell_h_m, 1.0])
    uv = T.cast(rays, calib, t, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), SPEC)
    centre = uv[:, :, 4]
    temps = np.full((24, 32), 15.0)
    temps[(np.floor(centre[..., 0]) == 2) & (np.floor(centre[..., 1]) == 4)] = 25.0
    temps[(np.floor(centre[..., 0]) == 1) & (np.floor(centre[..., 1]) == 7)] = 9.0
    cell_t, count, panel, n_px, _, _ = T.frame_readings(uv, temps, SPEC)
    assert count[4, 2] > 0 and cell_t[4, 2] == 25.0
    hot = {(r, c) for r, c in zip(*np.nonzero(np.abs(cell_t - panel) > T.HOTSPOT_DELTA_C))}
    assert hot == {(4, 2), (7, 1)}
