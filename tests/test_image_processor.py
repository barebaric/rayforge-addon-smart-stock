"""Tests for ImageProcessor class with synthetic images."""

import sys
from pathlib import Path

import pytest
import numpy as np

addon_path = Path(__file__).parent.parent
if str(addon_path) not in sys.path:
    sys.path.insert(0, str(addon_path))

from smart_stock.services.image_processor import (  # noqa: E402
    ImageProcessor,
    ProcessingConfig,
)


class TestProcessingConfig:
    """Tests for ProcessingConfig dataclass."""

    def test_default_config_instantiation(self):
        """Test that default config can be instantiated."""
        config = ProcessingConfig()
        assert config is not None

    def test_default_config_values(self):
        """Test that default config has expected values."""
        config = ProcessingConfig()
        assert config.clahe_clip_limit == 2.0
        assert config.clahe_tile_size == (8, 8)
        assert config.gaussian_blur_size == 5
        assert config.morphology_kernel_size == 5
        assert config.difference_threshold == 50

    def test_custom_config_values(self):
        """Test that custom config values can be set."""
        config = ProcessingConfig(
            clahe_clip_limit=4.0,
            clahe_tile_size=(4, 4),
            gaussian_blur_size=7,
            morphology_kernel_size=3,
            difference_threshold=20,
        )
        assert config.clahe_clip_limit == 4.0
        assert config.clahe_tile_size == (4, 4)
        assert config.gaussian_blur_size == 7
        assert config.morphology_kernel_size == 3
        assert config.difference_threshold == 20


class TestImageProcessorInstantiation:
    """Tests for ImageProcessor instantiation."""

    def test_default_instantiation(self):
        """Test that ImageProcessor can be instantiated with defaults."""
        processor = ImageProcessor()
        assert processor is not None
        assert processor.config is not None

    def test_instantiation_with_custom_config(self):
        """Test that ImageProcessor can be instantiated with custom config."""
        config = ProcessingConfig(clahe_clip_limit=3.0)
        processor = ImageProcessor(config=config)
        assert processor.config.clahe_clip_limit == 3.0

    def test_config_is_processing_config_instance(self):
        """Test that config is a ProcessingConfig instance."""
        processor = ImageProcessor()
        assert isinstance(processor.config, ProcessingConfig)


class TestNormalizeBrightnessContrast:
    """Tests for normalize_brightness_contrast method."""

    @pytest.fixture
    def processor(self):
        """Create a default ImageProcessor instance."""
        return ImageProcessor()

    @pytest.fixture
    def custom_processor(self):
        """Create an ImageProcessor with custom config."""
        config = ProcessingConfig(
            clahe_clip_limit=4.0,
            clahe_tile_size=(4, 4),
        )
        return ImageProcessor(config=config)

    def test_normalize_grayscale_image(self, processor):
        """Test normalization of grayscale image."""
        image = np.random.randint(50, 200, size=(100, 100), dtype=np.uint8)
        result = processor.normalize_brightness_contrast(image)

        assert result is not None
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_normalize_color_image(self, processor):
        """Test normalization of BGR color image."""
        image = np.random.randint(50, 200, size=(100, 100, 3), dtype=np.uint8)
        result = processor.normalize_brightness_contrast(image)

        assert result is not None
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_normalize_dark_image_brightness_increased(self, processor):
        """Test that dark images have brightness enhanced."""
        dark_image = np.full((100, 100, 3), 30, dtype=np.uint8)
        result = processor.normalize_brightness_contrast(dark_image)

        result_mean = np.mean(result)
        original_mean = np.mean(dark_image)

        assert result_mean > original_mean

    def test_normalize_bright_image_contrast_adjusted(self, processor):
        """Test that bright images get contrast adjustment."""
        bright_image = np.full((100, 100, 3), 220, dtype=np.uint8)
        result = processor.normalize_brightness_contrast(bright_image)

        assert result is not None
        assert result.shape == bright_image.shape

    def test_normalize_with_custom_config(self, custom_processor):
        """Test normalization with custom CLAHE parameters."""
        image = np.random.randint(50, 200, size=(100, 100), dtype=np.uint8)
        result = custom_processor.normalize_brightness_contrast(image)

        assert result is not None
        assert result.shape == image.shape

    def test_normalize_preserves_dimensions(self, processor):
        """Test that normalization preserves image dimensions."""
        for shape in [(50, 50), (100, 200), (200, 100, 3)]:
            image = np.random.randint(0, 255, size=shape, dtype=np.uint8)
            result = processor.normalize_brightness_contrast(image)
            assert result.shape == shape

    def test_normalize_various_image_sizes(self, processor):
        """Test normalization on various image sizes."""
        sizes = [(50, 50), (100, 100), (200, 300), (512, 512)]
        for height, width in sizes:
            image = np.random.randint(
                0, 255, size=(height, width), dtype=np.uint8
            )
            result = processor.normalize_brightness_contrast(image)
            assert result.shape == (height, width)

    def test_normalize_uniform_image(self, processor):
        """Test normalization of uniform image."""
        uniform_image = np.full((100, 100), 128, dtype=np.uint8)
        result = processor.normalize_brightness_contrast(uniform_image)

        assert result is not None
        assert result.shape == uniform_image.shape


class TestPrepareForComparison:
    """Tests for prepare_for_comparison method."""

    @pytest.fixture
    def processor(self):
        """Create a default ImageProcessor instance."""
        return ImageProcessor()

    def test_prepare_returns_grayscale(self, processor):
        """Test that prepare returns grayscale image."""
        image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        result = processor.prepare_for_comparison(image)

        assert len(result.shape) == 2
        assert result.dtype == np.uint8

    def test_prepare_preserves_dimensions(self, processor):
        """Test that prepare preserves spatial dimensions."""
        for height, width in [(100, 100), (200, 150), (300, 400)]:
            image = np.random.randint(
                0, 255, size=(height, width, 3), dtype=np.uint8
            )
            result = processor.prepare_for_comparison(image)
            assert result.shape == (height, width)

    def test_prepare_uniform_image(self, processor):
        """Test prepare on uniform image."""
        uniform_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = processor.prepare_for_comparison(uniform_image)

        assert result is not None
        assert result.shape == (100, 100)


class TestComputeDifference:
    """Tests for compute_difference method."""

    @pytest.fixture
    def processor(self):
        """Create a default ImageProcessor instance."""
        return ImageProcessor()

    def test_compute_difference_returns_binary_mask(self, processor):
        """Test that difference returns binary mask."""
        reference = np.zeros((200, 200, 3), dtype=np.uint8)
        current = np.zeros((200, 200, 3), dtype=np.uint8)
        current[50:150, 50:150] = [255, 255, 255]

        diff_mask = processor.compute_difference(current, reference)

        assert diff_mask is not None
        assert len(diff_mask.shape) == 2
        assert set(np.unique(diff_mask)).issubset({0, 255})

    def test_compute_difference_detects_new_object(self, processor):
        """Test that difference detects new objects."""
        reference = np.zeros((200, 200, 3), dtype=np.uint8)
        current = np.zeros((200, 200, 3), dtype=np.uint8)
        current[50:150, 50:150] = [255, 255, 255]

        diff_mask = processor.compute_difference(current, reference)

        assert np.any(diff_mask > 0)

    def test_compute_difference_empty_images(self, processor):
        """Test difference between two empty images."""
        reference = np.zeros((100, 100, 3), dtype=np.uint8)
        current = np.zeros((100, 100, 3), dtype=np.uint8)

        diff_mask = processor.compute_difference(current, reference)

        # Two empty images should have no difference
        assert diff_mask is not None
        assert np.sum(diff_mask > 0) == 0

    def test_compute_difference_preserves_dimensions(self, processor):
        """Test that difference preserves image dimensions."""
        for height, width in [(100, 100), (150, 200)]:
            reference = np.zeros((height, width, 3), dtype=np.uint8)
            current = np.random.randint(
                0, 255, size=(height, width, 3), dtype=np.uint8
            )

            diff_mask = processor.compute_difference(current, reference)
            assert diff_mask.shape == (height, width)

    def test_compute_difference_with_custom_threshold(self):
        """Test difference with custom threshold configuration."""
        config = ProcessingConfig(difference_threshold=50)
        processor = ImageProcessor(config=config)

        reference = np.zeros((200, 200, 3), dtype=np.uint8)
        current = np.zeros((200, 200, 3), dtype=np.uint8)
        current[50:150, 50:150] = [255, 255, 255]

        diff_mask = processor.compute_difference(current, reference)

        assert diff_mask is not None

    def test_compute_difference_same_images(self, processor):
        """Test difference between identical images."""
        image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)

        diff_mask = processor.compute_difference(image, image)

        # Same image should have no difference
        assert np.sum(diff_mask > 0) == 0

    def test_compute_difference_detects_color_change(self, processor):
        """Test that color changes are detected even with same brightness."""
        reference = np.full((200, 200, 3), 128, dtype=np.uint8)
        current = np.full((200, 200, 3), 128, dtype=np.uint8)
        current[50:150, 50:150] = [255, 0, 0]

        diff_mask = processor.compute_difference(current, reference)

        assert np.any(diff_mask > 0)

    def test_compute_difference_color_vs_grayscale_value(self, processor):
        """Test that pure color shift (same gray) is detected."""
        reference = np.zeros((200, 200, 3), dtype=np.uint8)
        reference[50:150, 50:150] = [85, 85, 85]

        current = np.zeros((200, 200, 3), dtype=np.uint8)
        current[50:150, 50:150] = [0, 128, 128]

        diff_mask = processor.compute_difference(current, reference)

        assert np.any(diff_mask > 0)


class TestApplyMorphology:
    """Tests for apply_morphology method."""

    @pytest.fixture
    def processor(self):
        """Create a default ImageProcessor instance."""
        return ImageProcessor()

    @pytest.fixture
    def sample_mask(self):
        """Create a sample binary mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        return mask

    def test_apply_morphology_clean(self, processor, sample_mask):
        """Test morphological clean operation."""
        result = processor.apply_morphology(sample_mask, operation="clean")

        assert result is not None
        assert result.shape == sample_mask.shape

    def test_apply_morphology_dilate(self, processor, sample_mask):
        """Test morphological dilate operation."""
        result = processor.apply_morphology(sample_mask, operation="dilate")

        assert result is not None
        # Dilated mask should have more or equal white pixels
        assert np.sum(result > 0) >= np.sum(sample_mask > 0)

    def test_apply_morphology_erode(self, processor, sample_mask):
        """Test morphological erode operation."""
        result = processor.apply_morphology(sample_mask, operation="erode")

        assert result is not None
        # Eroded mask should have fewer or equal white pixels
        assert np.sum(result > 0) <= np.sum(sample_mask > 0)

    def test_apply_morphology_default_operation(self, processor, sample_mask):
        """Test default morphological operation (clean)."""
        result = processor.apply_morphology(sample_mask)

        assert result is not None
        assert result.shape == sample_mask.shape

    def test_apply_morphology_empty_mask(self, processor):
        """Test morphology on empty mask."""
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        result = processor.apply_morphology(empty_mask, operation="clean")

        assert np.sum(result > 0) == 0

    def test_apply_morphology_full_mask(self, processor):
        """Test morphology on full mask."""
        full_mask = np.full((100, 100), 255, dtype=np.uint8)
        result = processor.apply_morphology(full_mask, operation="erode")

        assert result is not None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def processor(self):
        """Create a default ImageProcessor instance."""
        return ImageProcessor()

    def test_small_image(self, processor):
        """Test processing very small images."""
        small_image = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        result = processor.normalize_brightness_contrast(small_image)

        assert result is not None
        assert result.shape == (10, 10)

    def test_noisy_image(self, processor):
        """Test processing noisy images."""
        noisy_image = np.random.randint(
            0, 255, size=(100, 100, 3), dtype=np.uint8
        )
        result = processor.normalize_brightness_contrast(noisy_image)

        assert result is not None

    def test_high_contrast_image(self, processor):
        """Test processing high contrast images."""
        high_contrast = np.zeros((100, 100, 3), dtype=np.uint8)
        high_contrast[::2, ::2] = 255

        result = processor.normalize_brightness_contrast(high_contrast)

        assert result is not None


class TestIntegration:
    """Integration tests combining multiple operations."""

    @pytest.fixture
    def processor(self):
        """Create a default ImageProcessor instance."""
        return ImageProcessor()

    def test_full_processing_pipeline(self, processor):
        """Test full processing pipeline from raw image to mask."""
        # Create synthetic image with object
        current = np.zeros((200, 200, 3), dtype=np.uint8)
        current[50:150, 50:150] = [200, 200, 200]

        # Create reference (empty)
        reference = np.zeros((200, 200, 3), dtype=np.uint8)

        # Compute difference
        diff_mask = processor.compute_difference(current, reference)

        # Apply morphology
        clean_mask = processor.apply_morphology(diff_mask, operation="clean")

        assert clean_mask is not None
        assert clean_mask.shape == (200, 200)

    def test_processing_with_multiple_objects(self, processor):
        """Test processing image with multiple objects."""
        reference = np.zeros((300, 300, 3), dtype=np.uint8)

        current = np.zeros((300, 300, 3), dtype=np.uint8)
        current[30:80, 30:80] = [255, 255, 255]
        current[120:180, 120:180] = [200, 200, 200]
        current[200:250, 200:280] = [150, 150, 150]

        diff_mask = processor.compute_difference(current, reference)

        assert diff_mask is not None
        assert np.any(diff_mask > 0)
