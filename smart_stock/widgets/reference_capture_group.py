from gi.repository import Adw, Gtk
from blinker import Signal
from gettext import gettext as _
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple
import time
import logging

from ..models.reference_image import ReferenceImage
from ..utils import get_output_size

if TYPE_CHECKING:
    from ..services.reference_manager import ReferenceManager
    from rayforge.camera.controller import CameraController
    from rayforge.machine.models.machine import Machine

logger = logging.getLogger(__name__)


class ReferenceCaptureGroup(Adw.PreferencesGroup):
    def __init__(
        self, reference_manager: Optional["ReferenceManager"] = None, **kwargs
    ):
        super().__init__(
            title=_("Smart Stock Reference"),
            description=_("Capture reference image for stock detection"),
            **kwargs,
        )
        self._controllers: List["CameraController"] = []
        self._selected_controller: Optional["CameraController"] = None
        self._reference_images: Dict[str, ReferenceImage] = {}
        self._reference_manager = reference_manager
        self._machine: Optional["Machine"] = None

        self.reference_captured = Signal()
        self.reference_cleared = Signal()

        self._setup_ui()
        self._load_from_manager()

    def _load_from_manager(self):
        """Load existing references from the reference manager."""
        if self._reference_manager is None:
            return

        for camera_id in self._reference_manager.get_all_camera_ids():
            ref = self._reference_manager.get_reference(camera_id)
            if ref is not None:
                self._reference_images[camera_id] = ref

    def _setup_ui(self):
        self._camera_row = Adw.ActionRow(title=_("Selected Camera"))
        self._camera_row.set_subtitle(_("No camera selected"))
        self.add(self._camera_row)

        self._capture_row = Adw.ActionRow(
            title=_("Capture Reference"),
            subtitle=_("Capture current camera view as reference"),
        )
        self._capture_button = Gtk.Button(label=_("Capture"))
        self._capture_button.set_valign(Gtk.Align.CENTER)
        self._capture_button.connect("clicked", self._on_capture_clicked)
        self._capture_row.add_suffix(self._capture_button)
        self.add(self._capture_row)

        self._clear_row = Adw.ActionRow(
            title=_("Clear Reference"),
            subtitle=_("Remove stored reference image"),
        )
        self._clear_button = Gtk.Button(label=_("Clear"))
        self._clear_button.set_valign(Gtk.Align.CENTER)
        self._clear_button.add_css_class("destructive-action")
        self._clear_button.connect("clicked", self._on_clear_clicked)
        self._clear_row.add_suffix(self._clear_button)
        self.add(self._clear_row)

        self._status_row = Adw.ActionRow(title=_("Status"))
        self._status_row.set_subtitle(_("No reference captured"))
        self.add(self._status_row)

        self._update_sensitivity()

    def set_controllers(self, controllers: List["CameraController"]):
        self._controllers = controllers
        self._update_sensitivity()

    def set_selected_controller(self, controller: "CameraController"):
        logger.debug(f"Setting selected controller: {controller}")
        self._selected_controller = controller
        if controller and hasattr(controller, "config"):
            camera_name = getattr(controller.config, "name", _("Unknown"))
            self._camera_row.set_subtitle(camera_name)
        else:
            self._camera_row.set_subtitle(_("No camera selected"))
        self._update_sensitivity()
        self._update_status()

    def set_machine(self, machine: "Machine"):
        self._machine = machine

    def get_reference_image(self, camera_id: str) -> Optional[ReferenceImage]:
        return self._reference_images.get(camera_id)

    def get_all_references(self) -> Dict[str, ReferenceImage]:
        return self._reference_images.copy()

    def _update_sensitivity(self):
        has_controller = self._selected_controller is not None
        self._capture_button.set_sensitive(has_controller)

        camera_id = self._get_camera_id()
        has_reference = (
            camera_id is not None and camera_id in self._reference_images
        )
        self._clear_button.set_sensitive(has_reference)

    def _update_status(self):
        camera_id = self._get_camera_id()
        if camera_id and camera_id in self._reference_images:
            ref = self._reference_images[camera_id]
            elapsed = time.time() - ref.capture_timestamp
            if elapsed < 60:
                time_str = _("Just now")
            elif elapsed < 3600:
                minutes = int(elapsed / 60)
                time_str = _(f"{minutes} minute(s) ago")
            else:
                hours = int(elapsed / 3600)
                time_str = _(f"{hours} hour(s) ago")
            self._status_row.set_subtitle(
                _("Reference captured {time}").format(time=time_str)
            )
        else:
            self._status_row.set_subtitle(_("No reference captured"))

    def _get_camera_id(self) -> Optional[str]:
        if self._selected_controller and hasattr(
            self._selected_controller, "config"
        ):
            return getattr(self._selected_controller.config, "device_id", None)
        return None

    def _get_physical_area(
        self,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get the physical area matching the canvas (full axis extents)."""
        if self._machine is None:
            return None
        axis_extents = self._machine.axis_extents
        return ((0.0, 0.0), (float(axis_extents[0]), float(axis_extents[1])))

    def _on_capture_clicked(self, button: Gtk.Button):
        logger.debug("Capture button clicked")
        if not self._selected_controller:
            logger.warning("No controller selected")
            return

        camera_id = self._get_camera_id()
        if not camera_id:
            logger.warning("No camera_id available")
            return

        if self._selected_controller.image_data is None:
            logger.warning("No frame available from camera")
            return

        physical_area = self._get_physical_area()
        if physical_area is None:
            logger.warning("No physical area available - cannot capture")
            self._status_row.set_subtitle(
                _("Cannot capture: machine work area unknown")
            )
            return

        output_size = get_output_size(physical_area)

        aligned_frame = self._selected_controller.get_work_surface_image(
            output_size=output_size,
            physical_area=physical_area,
        )

        if aligned_frame is None:
            logger.warning("Failed to get world-aligned image")
            return

        logger.debug(
            f"Captured world-aligned frame with shape: {aligned_frame.shape}"
        )

        reference = ReferenceImage(
            raw_frame=aligned_frame.copy(),
            capture_timestamp=time.time(),
            camera_id=camera_id,
            physical_area=physical_area,
            output_size=output_size,
        )

        self._reference_images[camera_id] = reference
        self._update_sensitivity()
        self._update_status()

        logger.debug(f"Emitting reference_captured signal for {camera_id}")

        if self._reference_manager:
            logger.debug("Storing reference directly in manager")
            self._reference_manager.set_reference(camera_id, reference)

        self.reference_captured.send(
            self, reference=reference, camera_id=camera_id
        )
        logger.info(f"Reference captured for camera {camera_id}")

    def _on_clear_clicked(self, button: Gtk.Button):
        camera_id = self._get_camera_id()
        if not camera_id:
            return

        if camera_id in self._reference_images:
            del self._reference_images[camera_id]

            if self._reference_manager:
                self._reference_manager.clear_reference(camera_id)

            self._update_sensitivity()
            self._update_status()
            self.reference_cleared.send(self, camera_id=camera_id)
