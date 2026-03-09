"""Stock detection dialog for detecting stock items from camera feed."""

import logging
import time
from gettext import gettext as _
from typing import List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
from gi.repository import Adw, GLib, Gdk, GdkPixbuf, Gtk

from rayforge.core.geo import Geometry
from rayforge.core.stock_asset import StockAsset
from rayforge.core.stock import StockItem
from rayforge.ui_gtk.shared.patched_dialog_window import PatchedDialogWindow

from ..models.reference_image import ReferenceImage
from ..services.contour_detector import ContourDetector
from ..services.image_processor import ImageProcessor
from ..utils import get_output_size

if TYPE_CHECKING:
    from rayforge.camera.controller import CameraController
    from rayforge.doceditor.editor import DocEditor
    from rayforge.machine.models.machine import Machine
    from ..services.reference_manager import ReferenceManager

logger = logging.getLogger(__name__)


class StockDetectionDialog(PatchedDialogWindow):
    """Dialog for detecting stock items from a captured reference."""

    def __init__(
        self,
        camera_id: str,
        reference_manager: Optional["ReferenceManager"] = None,
        doc_editor: Optional["DocEditor"] = None,
        transient_for=None,
    ):
        super().__init__(
            transient_for=transient_for,
            skip_usage_tracking=True,
        )
        self._camera_id = camera_id
        self._reference_manager = reference_manager
        self._doc_editor = doc_editor
        self._controller: Optional["CameraController"] = None
        self._machine: Optional["Machine"] = None
        self._reference_image: Optional[ReferenceImage] = None
        self._current_frame_world: Optional[np.ndarray] = None
        self._detected_geometries: List[Geometry] = []
        self._capture_source_id: Optional[int] = None
        self._detecting = False
        self._last_sensitivity_state: Optional[bool] = None

        self._physical_area: Optional[
            Tuple[Tuple[float, float], Tuple[float, float]]
        ] = None
        self._output_size: Optional[Tuple[int, int]] = None

        self._processor = ImageProcessor()
        self._detector = ContourDetector()
        self._live_mask: Optional[np.ndarray] = None
        self._mask_accumulator: Optional[np.ndarray] = None

        self.set_title(_("Stock Detection"))
        self.set_default_size(1150, 750)

        self._setup_ui()
        self._load_reference()
        self._load_camera()
        self._connect_camera_signals()

    def _setup_ui(self):
        self._main_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
        )
        self.set_content(self._main_box)

        toolbar_view = Adw.ToolbarView()
        toolbar_view.set_vexpand(True)
        self._main_box.append(toolbar_view)

        header_bar = Adw.HeaderBar()
        toolbar_view.add_top_bar(header_bar)

        self._close_btn = Gtk.Button(label=_("Close"))
        self._close_btn.connect("clicked", self._on_close)
        header_bar.pack_start(self._close_btn)

        self._create_btn = Gtk.Button(label=_("Create Stock Items"))
        self._create_btn.add_css_class("suggested-action")
        self._create_btn.set_sensitive(False)
        self._create_btn.connect("clicked", self._on_create_clicked)
        header_bar.pack_end(self._create_btn)

        content_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=24,
            margin_start=24,
            margin_end=24,
            margin_top=12,
            margin_bottom=24,
        )
        toolbar_view.set_content(content_box)

        preview_frame = Gtk.Frame(
            hexpand=True,
            vexpand=True,
        )
        content_box.append(preview_frame)

        self._preview_drawing = Gtk.DrawingArea()
        self._preview_drawing.set_vexpand(True)
        self._preview_drawing.set_hexpand(True)
        self._preview_drawing.set_draw_func(self._draw_preview)
        preview_frame.set_child(self._preview_drawing)

        right_scroll = Gtk.ScrolledWindow()
        right_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        right_scroll.set_propagate_natural_width(False)
        right_scroll.set_propagate_natural_height(True)
        right_scroll.set_size_request(500, -1)
        right_scroll.set_hexpand(False)
        content_box.append(right_scroll)

        right_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
        )
        right_scroll.set_child(right_box)

        self._settings_group = Adw.PreferencesGroup(
            title=_("Detection Settings"),
        )
        right_box.append(self._settings_group)

        self._camera_row = Adw.ActionRow(
            title=_("Camera"),
            subtitle=_("No camera selected"),
        )
        self._camera_button = Gtk.Button(label=_("Change"))
        self._camera_button.set_valign(Gtk.Align.CENTER)
        self._camera_button.connect("clicked", self._on_camera_change_clicked)
        self._camera_row.add_suffix(self._camera_button)
        self._settings_group.add(self._camera_row)

        self._capture_row = Adw.ActionRow(
            title=_("Capture Reference"),
            subtitle=_("Capture current view as reference"),
        )
        self._capture_button = Gtk.Button(label=_("Capture"))
        self._capture_button.set_valign(Gtk.Align.CENTER)
        self._capture_button.connect("clicked", self._on_capture_clicked)
        self._capture_row.add_suffix(self._capture_button)
        self._settings_group.add(self._capture_row)

        self._threshold_row = Adw.ActionRow(
            title=_("Sensitivity"),
            subtitle=_("Lower values detect more changes"),
        )
        self._settings_group.add(self._threshold_row)

        self._threshold_scale = Gtk.Scale()
        self._threshold_scale.set_range(5, 200)
        self._threshold_scale.set_value(50)
        self._threshold_scale.set_hexpand(True)
        self._threshold_scale.set_draw_value(True)
        self._threshold_scale.set_value_pos(Gtk.PositionType.RIGHT)
        self._threshold_scale.connect(
            "value-changed", self._on_threshold_changed
        )
        self._threshold_row.add_suffix(self._threshold_scale)

        self._status_row = Adw.ActionRow(
            title=_("Status"),
            subtitle=_("Ready - Click 'Capture Reference' to begin"),
        )
        self._settings_group.add(self._status_row)

        self._detect_btn = Gtk.Button(label=_("Detect Stock"))
        self._detect_btn.set_sensitive(False)
        self._detect_btn.add_css_class("pill")
        self._detect_btn.connect("clicked", self._on_detect_stock)
        right_box.append(self._detect_btn)

        self._progress_bar = Gtk.ProgressBar(
            hexpand=True,
            valign=Gtk.Align.END,
            visible=False,
        )
        self._progress_bar.add_css_class("thin-progress-bar")
        self._apply_progress_bar_style()
        self._main_box.append(self._progress_bar)

    def _apply_progress_bar_style(self) -> None:
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(
            b"""
            progressbar.thin-progress-bar {
                min-height: 5px;
            }
            """
        )
        Gtk.StyleContext.add_provider_for_display(
            self.get_display(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _load_reference(self):
        if self._reference_manager is None:
            logger.warning("No reference manager provided")
            self._status_row.set_subtitle(_("No reference manager available"))
            return

        self._reference_image = self._reference_manager.get_reference(
            self._camera_id
        )

        if self._reference_image is not None:
            self._physical_area = self._reference_image.physical_area
            self._output_size = self._reference_image.output_size
            self._status_row.set_subtitle(_("Reference loaded - Ready"))
        else:
            self._status_row.set_subtitle(
                _("No reference - Click 'Capture Reference' to capture one")
            )

    def _load_camera(self):
        if self._doc_editor and hasattr(self._doc_editor, "context"):
            camera_mgr = self._doc_editor.context.camera_mgr
            self._machine = self._doc_editor.context.machine
        else:
            return

        for ctrl in camera_mgr.controllers:
            if ctrl.config.device_id == self._camera_id:
                self._controller = ctrl
                camera_name = getattr(ctrl.config, "name", self._camera_id)
                self._camera_row.set_subtitle(camera_name)
                break

        if self._controller is None:
            self._camera_row.set_subtitle(_("Not found"))
            self._show_error(
                _("Camera not found: {camera_id}").format(
                    camera_id=self._camera_id
                )
            )
            return

        if self._physical_area is None and self._machine is not None:
            axis_extents = self._machine.axis_extents
            self._physical_area = (
                (0.0, 0.0),
                (float(axis_extents[0]), float(axis_extents[1])),
            )

        if self._physical_area is not None and self._output_size is None:
            self._output_size = get_output_size(self._physical_area)

        self._update_button_sensitivity()

    def _connect_camera_signals(self):
        if self._capture_source_id:
            GLib.source_remove(self._capture_source_id)
            self._capture_source_id = None

        if self._controller:
            self._controller.subscribe()
            self._capture_source_id = GLib.timeout_add(
                33, self._on_capture_frame
            )

    def _on_capture_frame(self) -> bool:
        if (
            self._controller is None
            or self._controller.image_data is None
            or self._physical_area is None
            or self._output_size is None
        ):
            return True

        self._current_frame_world = self._controller.get_work_surface_image(
            output_size=self._output_size,
            physical_area=self._physical_area,
        )

        if (
            self._reference_image is not None
            and self._reference_image.raw_frame is not None
            and self._current_frame_world is not None
        ):
            ref_frame = self._reference_image.raw_frame
            raw_mask = self._processor.compute_difference(
                self._current_frame_world, ref_frame
            )
            self._live_mask = self._apply_mask_smoothing(raw_mask, alpha=0.3)
        else:
            self._live_mask = None
            self._mask_accumulator = None

        self._preview_drawing.queue_draw()
        self._update_button_sensitivity()
        return True

    def _apply_mask_smoothing(
        self, mask: np.ndarray, alpha: float = 0.3
    ) -> np.ndarray:
        if self._mask_accumulator is None:
            self._mask_accumulator = mask.astype(np.float32)
            return mask

        if self._mask_accumulator.shape != mask.shape:
            self._mask_accumulator = mask.astype(np.float32)
            return mask

        cv2.accumulateWeighted(mask, self._mask_accumulator, alpha)
        smoothed = self._mask_accumulator.astype(np.uint8)
        _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
        return smoothed

    def _on_detect_stock(self, button):
        if self._current_frame_world is None:
            self._show_error(_("No camera frame available"))
            return

        if self._reference_image is None:
            self._show_error(_("No reference image loaded"))
            return

        ref_frame = self._reference_image.raw_frame
        if ref_frame is None:
            self._show_error(_("Reference image has no frame data"))
            return

        if self._detecting:
            return

        if self._physical_area is None or self._output_size is None:
            self._show_error(_("No physical area or output size available"))
            return

        self._detecting = True
        self._detect_btn.set_sensitive(False)
        self._status_row.set_title(_("Processing..."))
        self._progress_bar.set_visible(True)
        self._progress_bar.pulse()

        current_frame_world = self._current_frame_world
        physical_area = self._physical_area

        if self._live_mask is not None:
            diff_mask = self._live_mask.copy()
            mask_nz = np.count_nonzero(diff_mask)
            logger.debug(
                f"Using smoothed live mask: non-zero pixels = {mask_nz}"
            )
        else:
            diff_mask = self._processor.compute_difference(
                current_frame_world, ref_frame
            )
            mask_nz = np.count_nonzero(diff_mask)
            logger.debug(f"Using fresh diff mask: non-zero pixels = {mask_nz}")

        def process_detection():
            try:
                logger.debug(
                    f"Reference frame shape: {ref_frame.shape}, "
                    f"current frame shape: {current_frame_world.shape}"
                )

                contours = self._detector.detect_contours(diff_mask)
                logger.debug(f"Found {len(contours)} raw contours")

                if not contours:
                    GLib.idle_add(self._on_detection_complete, False, 0)
                    return

                (x_min, y_min), (x_max, y_max) = physical_area
                img_height, img_width = current_frame_world.shape[:2]

                scale_x = (x_max - x_min) / img_width
                scale_y = (y_max - y_min) / img_height

                world_geometries = []
                for contour in contours:
                    if contour.points is None or len(contour.points) < 3:
                        continue

                    world_points = []
                    for point in contour.points:
                        px, py = point[0], point[1]
                        px = max(0.0, min(px, img_width - 1))
                        py = max(0.0, min(py, img_height - 1))
                        wx = x_min + px * scale_x
                        wy = y_max - py * scale_y
                        wx = max(x_min, min(wx, x_max))
                        wy = max(y_min, min(wy, y_max))
                        world_points.append((wx, wy))

                    geo = Geometry.from_points(world_points)
                    world_geometries.append(geo)

                self._detected_geometries = world_geometries

                GLib.idle_add(
                    self._on_detection_complete,
                    True,
                    len(self._detected_geometries),
                )

            except Exception as e:
                logger.error(f"Error detecting stock: {e}")
                GLib.idle_add(self._on_detection_error, str(e))

        GLib.idle_add(process_detection)

    def _on_detection_complete(self, success: bool, count: int):
        self._detecting = False
        self._progress_bar.set_visible(False)
        self._detect_btn.set_sensitive(True)

        if success and count > 0:
            self._status_row.set_title(
                _("Detected {count} items").format(count=count)
            )
            self._status_row.set_subtitle(
                _("Click 'Create Stock Items' to add them to the document")
            )
            self._create_btn.set_sensitive(True)
        else:
            self._status_row.set_title(_("No stock detected"))
            self._status_row.set_subtitle(
                _("Try adjusting the stock placement or lighting")
            )

        self._preview_drawing.queue_draw()

    def _on_detection_error(self, message: str):
        self._detecting = False
        self._progress_bar.set_visible(False)
        self._detect_btn.set_sensitive(True)
        self._status_row.set_title(_("Detection failed"))
        self._status_row.set_subtitle(message)
        self._show_error(message)

    def _draw_preview(self, area, cr, width, height):
        if self._current_frame_world is None:
            cr.set_source_rgb(0.1, 0.1, 0.1)
            cr.paint()
            return

        frame_rgb = cv2.cvtColor(self._current_frame_world, cv2.COLOR_BGR2RGB)

        frame_height, frame_width = frame_rgb.shape[:2]
        scale_x = width / frame_width
        scale_y = height / frame_height
        scale = min(scale_x, scale_y)

        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        frame_resized = cv2.resize(
            frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        frame_resized = np.ascontiguousarray(frame_resized)

        offset_x = (width - new_width) // 2
        offset_y = (height - new_height) // 2

        pixels = GLib.Bytes.new(frame_resized.tobytes())
        pb = GdkPixbuf.Pixbuf.new_from_bytes(
            pixels,
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            new_width,
            new_height,
            new_width * 3,
        )

        cr.set_source_rgb(0.0, 0.0, 0.0)
        cr.paint()

        Gdk.cairo_set_source_pixbuf(cr, pb, offset_x, offset_y)
        cr.paint()

        if self._live_mask is not None:
            mask_rgb = cv2.cvtColor(self._live_mask, cv2.COLOR_GRAY2RGB)
            mask_resized = cv2.resize(
                mask_rgb,
                (new_width, new_height),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_resized = np.ascontiguousarray(mask_resized)
            mask_pixels = GLib.Bytes.new(mask_resized.tobytes())
            mask_pb = GdkPixbuf.Pixbuf.new_from_bytes(
                mask_pixels,
                GdkPixbuf.Colorspace.RGB,
                False,
                8,
                new_width,
                new_height,
                new_width * 3,
            )
            cr.set_source_rgba(1.0, 0.0, 0.0, 0.5)
            Gdk.cairo_set_source_pixbuf(cr, mask_pb, offset_x, offset_y)
            cr.paint_with_alpha(0.5)

        if self._physical_area is None:
            return

        (x_min, y_min), (x_max, y_max) = self._physical_area
        world_to_preview_x = new_width / (x_max - x_min)
        world_to_preview_y = new_height / (y_max - y_min)

        for i, geometry in enumerate(self._detected_geometries):
            if geometry is None:
                continue

            cr.set_source_rgba(1.0, 0.0, 1.0, 0.8)
            cr.set_line_width(2.0)

            points = self._geometry_to_points(geometry)
            if points and len(points) > 0:
                first = True
                for wx, wy in points:
                    px = (wx - x_min) * world_to_preview_x + offset_x
                    py = (y_max - wy) * world_to_preview_y + offset_y
                    if first:
                        cr.move_to(px, py)
                        first = False
                    else:
                        cr.line_to(px, py)
                cr.close_path()
                cr.stroke()

                cr.set_source_rgba(0.0, 1.0, 0.0, 0.3)
                cr.fill_preserve()
                cr.stroke()

    def _geometry_to_points(
        self, geometry: Geometry
    ) -> List[Tuple[float, float]]:
        points = []
        for segment in geometry.segments():
            for point in segment:
                points.append((point[0], point[1]))
        return points

    def _on_create_clicked(self, button):
        if not self._detected_geometries:
            return

        if not self._doc_editor:
            self._show_error(_("No document editor available"))
            return

        doc = self._doc_editor.doc
        if not doc:
            self._show_error(_("No document available"))
            return

        created_count = 0
        for i, geometry in enumerate(self._detected_geometries):
            if geometry is None:
                continue

            rect = geometry.rect()
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            logger.debug(
                f"Stock {i + 1}: rect = ({rect[0]:.1f}, {rect[1]:.1f}, "
                f"{rect[2]:.1f}, {rect[3]:.1f}), "
                f"size = ({width:.1f}, {height:.1f})"
            )
            if width > 0 and height > 0:
                asset = StockAsset(
                    name=_("Stock {num}").format(num=i + 1),
                    geometry=geometry,
                )
                doc.add_asset(asset)

                item = StockItem(
                    stock_asset_uid=asset.uid,
                    name=_("Stock Item {num}").format(num=i + 1),
                )
                doc.add_child(item)
                item.set_size(width, height)
                item.pos = (rect[0], rect[1])
                logger.debug(
                    f"Stock {i + 1}: final pos={item.pos}, size={item.size}"
                )
                created_count += 1

        self._show_toast(
            _("Created {count} stock items").format(count=created_count)
        )
        self.close()

    def _update_button_sensitivity(self):
        has_reference = self._reference_image is not None
        has_camera = self._controller is not None
        has_frame = self._current_frame_world is not None
        should_be_sensitive = (
            has_reference and has_camera and has_frame and not self._detecting
        )

        if self._last_sensitivity_state != should_be_sensitive:
            logger.debug(
                f"Button sensitivity changed: {should_be_sensitive} "
                f"(ref={has_reference}, camera={has_camera}, "
                f"frame={has_frame}, detecting={self._detecting})"
            )
            self._last_sensitivity_state = should_be_sensitive

        self._detect_btn.set_sensitive(should_be_sensitive)
        self._capture_button.set_sensitive(has_camera and has_frame)

    def _show_error(self, message: str):
        self._status_row.set_title(_("Error"))
        self._status_row.set_subtitle(message)
        self._show_toast(message)

    def _show_toast(self, message: str):
        toast_overlay = self._main_box.get_ancestor(Adw.ToastOverlay)
        if toast_overlay:
            toast = Adw.Toast(title=message, timeout=3)
            toast_overlay.add_toast(toast)

    def _on_close(self, button):
        self._cleanup()
        self.close()

    def _on_threshold_changed(self, scale):
        threshold = int(scale.get_value())
        self._processor.set_threshold(threshold)

    def _on_capture_clicked(self, button):
        logger.debug("Capture button clicked")
        if not self._controller:
            logger.warning("No controller available")
            self._show_error(_("No camera available"))
            return

        if self._controller.image_data is None:
            logger.warning("No frame available from camera")
            self._show_error(_("No camera frame available"))
            return

        if self._machine is None:
            logger.warning("No machine available - cannot capture")
            self._show_error(_("Machine work area unknown"))
            return

        axis_extents = self._machine.axis_extents
        physical_area = (
            (0.0, 0.0),
            (float(axis_extents[0]), float(axis_extents[1])),
        )
        output_size = get_output_size(physical_area)

        aligned_frame = self._controller.get_work_surface_image(
            output_size=output_size,
            physical_area=physical_area,
        )

        if aligned_frame is None:
            logger.warning("Failed to get world-aligned image")
            self._show_error(_("Failed to capture aligned image"))
            return

        logger.debug(
            f"Captured world-aligned frame with shape: {aligned_frame.shape}"
        )

        reference = ReferenceImage(
            raw_frame=aligned_frame.copy(),
            capture_timestamp=time.time(),
            camera_id=self._camera_id,
            physical_area=physical_area,
            output_size=output_size,
        )

        self._reference_image = reference
        self._physical_area = physical_area
        self._output_size = output_size

        if self._reference_manager:
            self._reference_manager.set_reference(self._camera_id, reference)

        self._status_row.set_subtitle(_("Reference captured - Ready"))
        self._update_button_sensitivity()
        logger.info(f"Reference captured for camera {self._camera_id}")

    def _on_camera_change_clicked(self, button):
        from rayforge.ui_gtk.camera.selection_dialog import (
            CameraSelectionDialog,
        )

        dialog = CameraSelectionDialog(self, mode="configured")
        dialog.present()

        def on_response(dialog, response_id):
            if response_id == "select" and dialog.selected_device_id:
                new_camera_id = dialog.selected_device_id
                if new_camera_id != self._camera_id:
                    self._switch_camera(new_camera_id)
            dialog.destroy()

        dialog.connect("response", on_response)

    def _switch_camera(self, new_camera_id: str):
        self._cleanup()
        self._camera_id = new_camera_id
        self._controller = None
        self._current_frame_world = None
        self._live_mask = None
        self._mask_accumulator = None
        self._detected_geometries = []
        self._create_btn.set_sensitive(False)
        self._load_reference()
        self._load_camera()
        self._connect_camera_signals()

    def _cleanup(self):
        if self._capture_source_id:
            GLib.source_remove(self._capture_source_id)
            self._capture_source_id = None

        if self._controller:
            self._controller.unsubscribe()

    def do_close_request(self, *args) -> bool:
        self._cleanup()
        return False
