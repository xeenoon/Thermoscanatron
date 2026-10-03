"""Panel profile: the cell lattice that labels, training targets and the tracker all share.

Panel coordinates: u across the columns (0..cols), v down the rows (0..rows), in cell units, origin at the
corner of cell (0, 0). A homography H maps panel (u, v, 1) to image pixels. The cell area is the lattice
inside the frame; the frame itself is outside it.

Mono panels with chamfered wafers show a bright diamond where four wafer corners meet. With half-cut cells
those sit on every second row boundary only, which fixes the row parity (and, for an odd row count, which
end is up) even when a single boundary is in view.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PanelSpec:
    name: str = "mono-36-halfcut"
    cols: int = 4
    rows: int = 9
    # Row boundaries (v = k) that carry diamonds, counted from row 0's top edge.
    diamond_rows: tuple[int, ...] = (2, 4, 6, 8)
    # Gridline (inter-cell gap) width as a fraction of a cell's width, for rendering line targets.
    line_frac: float = 0.03
    # Physical cell pitch in metres (column width, row height). Only sets the panel's distance from the camera,
    # which matters for the thermal camera's few-cm parallax; the row/column ratio is measured from video.
    cell_w_m: float = 0.156
    cell_h_m: float = 0.108

    @property
    def diamond_points(self) -> list[tuple[float, float]]:
        """Panel coordinates of every diamond: inner column boundaries x diamond rows."""
        return [(float(u), float(v)) for v in self.diamond_rows for u in range(1, self.cols)]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | None) -> "PanelSpec":
        if path is None:
            return cls()
        d = json.loads(Path(path).read_text())
        d["diamond_rows"] = tuple(d["diamond_rows"])
        return cls(**d)


DEFAULT = PanelSpec()
