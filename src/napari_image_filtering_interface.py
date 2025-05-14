"""
Napari Plugin Configuration and Widget

This module sets up the Napari plugin interface for image filtering.
"""
import napari
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton,
    QLabel, QSlider, QHBoxLayout
)
from qtpy.QtCore import Qt
from .napari_image_filters import (
    apply_grayscale, apply_saturation,
    apply_edge_enhance, apply_edge_detection,
    apply_gaussian_blur, apply_contrast_enhancement,
    apply_texture_analysis, apply_adaptive_threshold,
    apply_sharpening, apply_ridge_detection, otsu_thresholding,
    otsu_thresholding_no_mask, split_channels
)

from .chat_interface import ChatWidget
from .pipelines import WorkflowWidget
from .cellpose_launch import CellposeLaunchPoint
from .napari_image_filters_ui import ImageFiltersUI


class ImageFilterWidget(QWidget):
    """
    Custom Napari widget for image filtering operations.

    Provides interactive controls for:
    - Grayscale conversion
    - Saturation adjustment
    - Edge enhancement
    - Edge detection
    """

    def __init__(self, viewer: napari.Viewer):
        """
        Initialize the image filter widget.

        Args:
            viewer (napari.Viewer): The Napari viewer instance
        """
        super().__init__()
        self.viewer = viewer
        self.original_data = None

        # History stack for undo functionality
        self.history_stack = []
        # Maximum number of undo steps
        self.max_history = 20

        # Create main layout
        layout = QVBoxLayout()

        # Add button layout at the top
        button_layout = QHBoxLayout()

        # Add Undo button
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self._undo_last_change)
        self.undo_button.setEnabled(False)  # Disabled initially
        button_layout.addWidget(self.undo_button)

        # Add layout to main layout
        layout.addLayout(button_layout)

        # Saturation slider
        sat_layout = QHBoxLayout()
        self.sat_label = QLabel("Saturation: 1.0")
        self.sat_slider = QSlider(Qt.Horizontal)
        self.sat_slider.setMinimum(0)
        self.sat_slider.setMaximum(200)
        self.sat_slider.setValue(100)
        self.sat_slider.valueChanged.connect(self._update_saturation)
        sat_layout.addWidget(self.sat_label)
        sat_layout.addWidget(self.sat_slider)
        layout.addLayout(sat_layout)

        # Filter buttons
        self.filter_buttons = [
            ("Split channels", self._split_channels),
            ("Grayscale", self._apply_grayscale),
            ("Edge Enhance", self._apply_edge_enhance),
            ("Edge Detection", self._apply_edge_detection),
            ("Gaussian Blur", self._apply_gaussian_blur),
            ("Contrast Enhance", self._apply_contrast_enhancement),
            ("Texture Analysis", self._apply_texture_analysis),
            ("Adaptive Threshold", self._apply_adaptive_threshold),
            ("Sharpen", self._apply_sharpening),
            ("Ridge Detection", self._apply_ridge_detection),
            ("Otsu thresholding", self._apply_otsu_thresholding),
            ("Otsu thresholding (no mask)", self._apply_otsu_thresholding_no_mask),
        ]

        # for name, method in self.filter_buttons:
        #     btn = QPushButton(name)
        #     btn.clicked.connect(method)
        #     layout.addWidget(btn)

        open_btn = QPushButton("View Image Filters")
        open_btn.clicked.connect(self._view_filters)
        layout.addWidget(open_btn)

        # Initialize the AI view
        self.chat_widget = ChatWidget(viewer, self)
        layout.addWidget(self.chat_widget)

        # Initialize the history view
        self.workflow = WorkflowWidget(viewer, self)
        layout.addWidget(self.workflow)

        # Initialize the cellpose interface, which is used to fine tune the
        # model which is the final stage in their custom pipeline
        self.model_interf = CellposeLaunchPoint(viewer, self)
        layout.addWidget(self.model_interf)

        self.setLayout(layout)

    def _view_filters(self):
        filters_interface = ImageFiltersUI(self.filter_buttons)
        filters_interface.exec_()

    def _get_current_layer(self):
        """
        Retrieve the currently selected image layer.

        Returns:
            napari.layers.Image or None: Current image layer

        Raises:
            ValueError: If no image layer is selected
        """
        # Get the selected layers
        selected_layers = self.viewer.layers.selection

        # Check if any layers are selected
        if not selected_layers:
            # pdb.set_trace()
            self.chat_widget.add_to_chat(
                "[ERROR] please select an image layer")
            raise ValueError("Please select an image layer")

        # Find the first image layer
        for layer in selected_layers:
            if isinstance(layer, napari.layers.Image):
                # Store original data (The first time called)
                if self.original_data is None:
                    self.original_data = layer.data.copy()
                return layer

        # If no image layer is found
        raise ValueError("Please select an image layer")

    def _update_saturation(self, value):
        """
        Update saturation label based on slider position.

        Args:
            value (int): Slider value (0-200)
        """
        # Convert slider value to saturation multiplier
        sat_value = value / 100.0
        self.sat_label.setText(f"Saturation: {sat_value:.2f}")
        self._apply_saturation()

    def _push_to_history(self, layer):
        """
        Push current image state to history stack.

        Args:
            layer (napari.layers.Image): Current image layer
        """
        if len(self.history_stack) >= self.max_history:
            # Remove oldest history item if we exceed max history
            self.history_stack.pop(0)

        # Store a copy of the current state
        self.history_stack.append(layer.data.copy())

        # Enable undo button when we have history
        self.undo_button.setEnabled(True)

    def _undo_last_change(self):
        """
        Undo the last change by restoring the previous state.
        """
        try:
            layer = self._get_current_layer()

            if self.history_stack:
                # Restore the previous state
                layer.data = self.history_stack.pop()

                # Disable undo button if no more history
                if not self.history_stack:
                    self.undo_button.setEnabled(False)

        except Exception as e:
            print(f"Error undoing last change: {e}")
            import traceback
            traceback.print_exc()

    def _apply_filter(self, filter_func):
        """
        Apply a filter to the current image layer.

        Args:
            filter_func (callable): Filter function to apply
        """
        try:
            # Get current layer
            layer = self._get_current_layer()
            # Store current state in history before applying filter
            self._push_to_history(layer)
            # Create copy to avoid modifying original
            original_data = layer.data.copy()

            # Preprocess for filters requiring grayscale input
            if filter_func in [
                    apply_grayscale,
                    apply_texture_analysis,
                    apply_adaptive_threshold,
                    otsu_thresholding,
                    otsu_thresholding_no_mask]:

                # RGB & RGBA images
                if original_data.ndim == 3 and original_data.shape[2] in [
                        3, 4]:
                    original_data = np.mean(original_data, axis=2)

                # Single-channel 3D array
                elif original_data.ndim == 3 and original_data.shape[2] == 1:
                    original_data = original_data.squeeze()

            # Handle multi-dimensional data
            if filter_func in [apply_edge_enhance, apply_edge_detection]:
                # Check if input is 3D stack or single image (2D)
                if original_data.ndim == 3 and original_data.shape[-1] not in [
                        3, 4]:
                    filtered_slices = []
                    for i in range(original_data.shape[0]):
                        # Apply filter to individual 2D slices
                        slice_data = original_data[i]
                        filtered_slice = filter_func(slice_data)
                        filtered_slices.append(np.asarray(filtered_slice))
                    # Rebuild 3D stack from processed slices
                    filtered_array = np.stack(filtered_slices)
                else:
                    # Process single 2D image or RGB image directly
                    filtered_array = filter_func(original_data)
            else:

                # special case: splitting into 3 channels, so add 3 new layers
                if filter_func is split_channels:
                    img_r, img_b, img_g = filter_func(original_data)
                    filter_name = filter_func.__name__.replace(
                        "apply_", "").replace("_", " ").title()
                    new_layer_name = f"{layer.name} | {filter_name}"

                    self.viewer.add_image(img_r, name=new_layer_name + '_r')
                    self.viewer.add_image(img_g, name=new_layer_name + '_g')
                    self.viewer.add_image(img_b, name=new_layer_name + '_b')
                    self.workflow.add_event_to_workflow(filter_func)
                    return

                # Apply filter directly for non-special cases
                filtered_array = filter_func(original_data)

            # Rename filter function to saturation for display
            if filter_func.__name__ == "<lambda>":
                filter_func.__name__ = "apply_saturation"

            # Create new layer with descriptive name and add to napari viewer
            filter_name = filter_func.__name__.replace(
                "apply_", "").replace("_", " ").title()
            new_layer_name = f"{layer.name} | {filter_name}"

            self.viewer.add_image(filtered_array, name=new_layer_name)
            self.workflow.add_event_to_workflow(filter_func)

        except Exception as e:
            print(f"Error applying filter: {e}")
            import traceback
            traceback.print_exc()

    def add_to_chat(self, log: str):
        self.chat_widget.add_to_chat(log)

    def _apply_saturation(self):
        """Apply saturation adjustment to current layer."""
        sat_value = self.sat_slider.value() / 100.0
        # Connect the saturation button to the filter method
        self._apply_filter(lambda img: apply_saturation(img, sat_value))

    def _apply_grayscale(self):
        """Apply grayscale filter to current layer."""
        self._apply_filter(apply_grayscale)

    def _apply_edge_enhance(self):
        """Apply edge enhancement to current layer."""
        self._apply_filter(apply_edge_enhance)

    def _apply_edge_detection(self):
        """Apply edge detection to current layer."""
        self._apply_filter(apply_edge_detection)

    def _apply_gaussian_blur(self):
        """Apply Gaussian blur to the current image"""
        self._apply_filter(apply_gaussian_blur)

    def _apply_contrast_enhancement(self):
        """Apply contrast enhancement to the current image"""
        self._apply_filter(apply_contrast_enhancement)

    def _apply_texture_analysis(self):
        """Apply texture analysis to the current image"""
        self._apply_filter(apply_texture_analysis)

    def _apply_adaptive_threshold(self):
        """Apply adaptive threshold to the current image"""
        self._apply_filter(apply_adaptive_threshold)

    def _apply_sharpening(self):
        """Apply sharpening filter to the current image."""
        self._apply_filter(apply_sharpening)

    def _apply_ridge_detection(self):
        """Apply ridge detection to the current image."""
        self._apply_filter(apply_ridge_detection)

    def _apply_otsu_thresholding(self):
        """Otsu thresholding filter for cell segmentation (with 5x5 ones kernel applied)"""
        self._apply_filter(otsu_thresholding)

    def _apply_otsu_thresholding_no_mask(self):
        self._apply_filter(otsu_thresholding_no_mask)

    def _split_channels(self):
        self._apply_filter(split_channels)


def napari_experimental_provide_dock_widget():
    """
    Napari plugin hook to provide the dock widget.

    Returns:
        callable: Function that creates the widget
    """
    def _widget_factory(viewer: napari.Viewer):
        return ImageFilterWidget(viewer)
    return _widget_factory
