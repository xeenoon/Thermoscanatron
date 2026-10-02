"""torch.export -> ExecuTorch .pte (XNNPACK CPU backend), plus a desktop runner for parity checks."""

from pathlib import Path

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime


def export_pte(model: torch.nn.Module, example: torch.Tensor, out_path: Path) -> Path:
    model = model.eval()
    exported = torch.export.export(model, (example,))
    program = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()]).to_executorch()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return out_path


class PteRunner:
    def __init__(self, path: Path):
        self.method = Runtime.get().load_program(path).load_method("forward")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.method.execute([x])[0]
