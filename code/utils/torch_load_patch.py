"""
Patches `torch.load` to default to `weights_only=False`.

torch>=2.6 changed the default to `weights_only=True`, which refuses to
unpickle non-tensor objects (e.g. `mmengine.logging.history_buffer.HistoryBuffer`)
found in mmdet/mmseg checkpoints. Import this module before any code path
that calls `torch.load` (directly or via mmcv/mmdet/mmseg/mmengine) so the
patch is in effect first.
"""
import torch

_real_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)


torch.load = _patched_torch_load
