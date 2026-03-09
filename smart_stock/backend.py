"""Backend module for smart stock addon (no UI dependencies)."""

from rayforge.core.hooks import hookimpl


@hookimpl
def on_unload():
    """Cleanup when addon is unloaded."""
    pass
