"""
CellViT-plus-plus's `cellvit.utils.logger.Logger.create_logger()` grabs the logger named
`__main__`, hardcodes its level to `DEBUG` regardless of the `level="INFO"` it was constructed
with (see `CellViTInference._instantiate_logger()`), and leaves `propagate` at its default
`True`. Combined with this pipeline's root logger (configured in `logging_utils.py`), that
causes two problems:

  1. Every CellViT log line is duplicated: once through CellViT's own StreamHandler (format
     `%(asctime)s [%(levelname)s] - message`, no logger name) and once more via propagation to
     the root logger's handler (format `... __main__: message`).
  2. DEBUG-level internals never meant to reach the console at CellViT's own requested
     `level="INFO"` ("Padding Tile", "Found invalid polygon - Fixing with buffer 0") flood the
     log, since the logger's effective level was forced to DEBUG independent of that request.

Patches `create_logger` to set the logger's level to what was actually requested (`self.level`)
instead of the hardcoded `"DEBUG"`, and to stop it propagating to the root logger, so CellViT's
own handler remains the single source of its log lines. Call `apply()` before constructing any
`CellViTInferenceMemory` - same "patch before use" convention as the other `*_patch.py` modules
in this package.
"""
_applied = False


def apply() -> None:
    global _applied
    if _applied:
        return
    from cellvit.utils.logger import Logger

    original_create_logger = Logger.create_logger

    def _patched_create_logger(self):
        logger = original_create_logger(self)
        logger.setLevel(self.level)
        logger.propagate = False
        return logger

    Logger.create_logger = _patched_create_logger
    _applied = True
