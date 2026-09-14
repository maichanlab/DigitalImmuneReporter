"""
Patches `pathopatch`'s pydantic model classes (used by CellViT-plus-plus's
`LivePatchWSIConfig`/`LivePatchWSIDataset`/`LivePatchWSIDataloader`) for pydantic v2.

`pathopatch` was written against pydantic v1, where a bare `field: Optional[X]` (no `= None`)
implicitly defaults to `None`. Pydantic v2 removed that implicit default — the same
`Optional[X]` annotation is now a *required* (but nullable) field — so constructing
`LivePatchWSIConfig` without explicitly passing every such field (e.g. `target_mag`, `level`,
`annotation_path`, `label_map`, ...) raises a `pydantic_core.ValidationError` ("Field
required"), even though every one of those fields is documented as optional and meant to
default to `None`.

Rather than editing the installed `pathopatch` package in place (fragile across reinstalls),
this walks its model classes' fields at runtime and gives every bare-`Optional` field an
explicit `None` default — same monkey-patch-before-use approach as `torch_load_patch.py` /
`slidevips_mpp_patch.py`. Call `apply()` before constructing any `pathopatch` model (i.e.
before `cellvit_utils.detect_cells_binary`'s call into `CellViTInferenceMemory.process_wsi`).
"""
import typing

_PATCHED_MODEL_PATHS = [
    ("pathopatch.patch_extraction.dataset", "LivePatchWSIConfig"),
]

_applied = False


def _default_to_none_for_bare_optional_fields(model_cls) -> None:
    import pydantic_core

    changed = False
    for field in model_cls.model_fields.values():
        if field.default is not pydantic_core.PydanticUndefined:
            continue  # already has an explicit default (or isn't Optional) - leave as-is
        if type(None) not in typing.get_args(field.annotation):
            continue  # not an Optional[...] field, a genuinely required field - leave as-is
        field.default = None
        changed = True
    if changed:
        model_cls.model_rebuild(force=True)


def apply() -> None:
    global _applied
    if _applied:
        return
    for module_path, class_name in _PATCHED_MODEL_PATHS:
        import importlib

        module = importlib.import_module(module_path)
        _default_to_none_for_bare_optional_fields(getattr(module, class_name))
    _applied = True
