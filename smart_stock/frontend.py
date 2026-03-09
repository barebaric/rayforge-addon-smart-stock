"""Frontend module for smart stock addon."""

import logging
from gettext import gettext as _

from gi.repository import Gio

from rayforge.context import get_context
from rayforge.core.hooks import hookimpl
from rayforge.ui_gtk.action_registry import MenuPlacement
from .dialogs.stock_detection_dialog import StockDetectionDialog
from .services.reference_manager import ReferenceManager

logger = logging.getLogger(__name__)

ADDON_NAME = "smart_stock"

_reference_manager: ReferenceManager | None = None


def get_reference_manager() -> ReferenceManager:
    """Get or create the singleton reference manager."""
    global _reference_manager
    if _reference_manager is None:
        _reference_manager = ReferenceManager()
        logger.debug(f"Created new ReferenceManager: {id(_reference_manager)}")
    else:
        logger.debug(
            f"Returning existing ReferenceManager: {id(_reference_manager)}"
        )
    return _reference_manager


@hookimpl
def register_actions(action_registry):
    """Register action for stock detection with menu placement."""
    action = Gio.SimpleAction.new("smart_stock_detect", None)

    def on_activate(action, param):
        logger.debug("smart_stock_detect action activated")
        manager = get_reference_manager()
        window = action_registry.window

        camera_mgr = get_context().camera_mgr
        controllers = camera_mgr.controllers

        if not controllers:
            return

        camera_id = controllers[0].config.device_id
        dialog = StockDetectionDialog(
            camera_id=camera_id,
            reference_manager=manager,
            doc_editor=window.doc_editor,
            transient_for=window,
        )
        dialog.present()

    action.connect("activate", on_activate)
    action_registry.register(
        action_name="smart_stock_detect",
        action=action,
        addon_name=ADDON_NAME,
        label=_("Detect Stock from Camera..."),
        menu=MenuPlacement(menu_id="tools", priority=60),
    )


@hookimpl
def on_unload():
    """Cleanup when addon is unloaded."""
    pass
