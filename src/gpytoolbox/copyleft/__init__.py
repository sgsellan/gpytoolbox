from .mesh_boolean import mesh_boolean
from .lazy_cage import lazy_cage
from .do_meshes_intersect import do_meshes_intersect


# swept_volume no longer relies on copyleft (GPL) code; it now lives in the main
# (MIT-licensed) gpytoolbox namespace. It remains importable from here for
# backwards compatibility, but this alias is deprecated and will be removed in
# gpytoolbox 0.4.0. The lazy __getattr__ below ensures the deprecation warning is
# only raised if a user actually accesses gpytoolbox.copyleft.swept_volume, and
# not when merely importing the copyleft module or its other functions.
def __getattr__(name):
    if name == "swept_volume":
        import warnings
        warnings.warn(
            "Importing swept_volume from gpytoolbox.copyleft is deprecated and "
            "will be removed in gpytoolbox 0.4.0. swept_volume is now MIT-licensed; "
            "import it from the main module instead: "
            "`from gpytoolbox import swept_volume`.",
            DeprecationWarning,
            stacklevel=2,
        )
        from gpytoolbox.swept_volume import swept_volume
        return swept_volume
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")