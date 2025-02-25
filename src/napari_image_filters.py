"""
Napari Plugin for Image Filtering and Processing

This module provides a set of image processing functions
that can be applied to images in the Napari viewer.

Key Features:
- Grayscale conversion
- Image cropping
- Saturation adjustment
- Edge enhancement
- Edge detection
"""

from typing import Union, Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from skimage.feature import local_binary_pattern
from scipy.ndimage import gaussian_filter
from cellotype.predict import CelloTypePredictor
from skimage import io


def apply_gaussian_blur(
    img: Union[Image.Image, np.ndarray, str], radius: float = 2.0
) -> np.ndarray:
    """
    Apply Gaussian blur to the image with adjustable radius.

    Args:
        img (PIL.Image, numpy.ndarray, or str): Input image
        radius (float): Blur radius (sigma value)

    Returns:
        numpy.ndarray: Blurred image
    """
    if radius < 0:
        raise ValueError

    pil_img = ensure_pil_image(img)
    img_array = np.array(pil_img)

    # Apply gaussian blur
    if len(img_array.shape) == 3:
        # For RGB images, apply to each channel
        blurred = np.zeros_like(img_array)
        for i in range(img_array.shape[2]):
            blurred[:, :, i] = gaussian_filter(
                img_array[:, :, i], sigma=radius)
    else:
        # For grayscale images
        blurred = gaussian_filter(img_array, sigma=radius)

    return blurred


def apply_contrast_enhancement(
    img: Union[Image.Image, np.ndarray, str], factor: float = 1.5
) -> np.ndarray:
    """
    Enhance image contrast using adaptive histogram equalization.

    Args:
        img (PIL.Image, numpy.ndarray, str): Input image
        factor (float): Contrast enhancement factor

    Returns:
        numpy.ndarray: Contrast-enhanced image
    """
    pil_img = ensure_pil_image(img)
    img_array = np.array(pil_img)

    # Convert to LAB color space for better contrast enhancement
    if len(img_array.shape) == 3:
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=factor, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Merge channels and convert back to RGB
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    else:
        # For grayscale images
        clahe = cv2.createCLAHE(clipLimit=factor, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_array)

    return enhanced


def apply_texture_analysis(
    img: Union[Image.Image, np.ndarray, str], radius: int = 3, n_points: int = 8
) -> np.ndarray:
    """
    Apply Local Binary Pattern for texture analysis.

    Args:
        img (PIL.Image, numpy.ndarray, str): Input image
        radius (int): Radius of the pattern
        n_points (int): Number of points in the pattern

    Returns:
        numpy.ndarray: Texture pattern image
    """
    pil_img = ensure_pil_image(img)
    input_array = np.array(pil_img)
    gray_img = sanitize_dimensional_image(pil_img)

    # Convert to uint8 before LBP computation
    if gray_img.dtype.kind == "f":
        gray_img = (gray_img * 255).clip(0, 255).astype(np.uint8)
    elif gray_img.dtype != np.uint8:
        gray_img = gray_img.astype(np.uint8)

    # Apply LBP
    lbp = local_binary_pattern(gray_img, n_points, radius, method="uniform")

    # Normalize to 0-255 range
    lbp_normalized = ((lbp - lbp.min()) *
                      (255.0 / (lbp.max() - lbp.min()))).astype(np.uint8)

    # If the input was a 3D array, maintain the shape for consistency
    if input_array.ndim == 3:
        return lbp_normalized[..., np.newaxis]

    return lbp_normalized


def apply_adaptive_threshold(
    img: Union[Image.Image, np.ndarray, str], block_size: int = 11, c: int = 2
) -> np.ndarray:
    """
    Apply adaptive thresholding for improved edge detection.

    Args:
        img (PIL.Image, numpy.ndarray, str): Input image
        block_size (int): Size of pixel neighborhood (must be odd)
        c (int): Constant subtracted from mean

    Returns:
        numpy.ndarray: Thresholded image
    """
    if block_size <= 0 or block_size % 2 == 0:
        raise ValueError
    pil_img = ensure_pil_image(img)
    input_array = np.array(pil_img)
    gray_img = sanitize_dimensional_image(pil_img)

    # Ensure block_size is odd
    block_size = block_size if block_size % 2 == 1 else block_size + 1

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c)

    if input_array.ndim == 3:
        return thresh[..., np.newaxis]

    return thresh


def apply_sharpening(img: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
    """
    Apply sharpening using a convolutional kernel.

    Args:
        img (PIL.Image, numpy.ndarray, str): Input image.

    Returns:
        numpy.ndarray: Sharpened image.
    """
    # Convert input to numpy array
    pil_img = ensure_pil_image(img)
    img_array = np.array(pil_img).astype(np.float32)

    # Define sharpening kernel
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]).astype(np.float32)

    # Apply filter using OpenCV, ensuring src is the converted numpy array
    sharpened = cv2.filter2D(img_array, -1, sharpen_kernel)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_ridge_detection(
        img: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
    """
    Apply ridge detection using a convolutional kernel.

    Args:
        img (PIL.Image, numpy.ndarray, str): Input image.

    Returns:
        numpy.ndarray: With ridge detection.
    """
    pil_img = ensure_pil_image(img)
    img_array = np.array(pil_img)

    ridge_kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    return np.clip(cv2.filter2D(img_array, -1, ridge_kernel),
                   0, 255).astype(np.uint8)


def ensure_pil_image(img: Union[Image.Image, np.ndarray, str]) -> Image.Image:
    """
    Convert input to PIL Image, handling different input types.

    Args:
        img (PIL.Image, numpy.ndarray, or str): Input image in various formats

    Returns:
        PIL.Image: Converted image

    Raises:
        TypeError: If input cannot be converted to PIL Image
    """
    # If already a PIL Image, return as-is
    if isinstance(img, Image.Image):
        return img

    # If string, open as image file
    if isinstance(img, str):
        return Image.open(img)

    # If NumPy array
    if isinstance(img, np.ndarray):
        # Handle different array shapes and data types
        # Ensure the array is uint8 to avoid potential conversion issues
        img_array = img.astype(np.uint8)

        # Handle different array dimensions
        if img_array.ndim == 2:
            # Grayscale image
            return Image.fromarray(img_array, mode="L")
        elif img_array.ndim == 3:
            # Check color channels
            if img_array.shape[2] == 3:
                # RGB image
                return Image.fromarray(img_array, mode="RGB")
            elif img_array.shape[2] == 4:
                # RGBA image
                return Image.fromarray(img_array, mode="RGBA")
            else:
                # Unusual channel count, try to convert to RGB
                if img_array.shape[2] == 1:
                    # Single channel, convert to grayscale
                    return Image.fromarray(img_array.squeeze(), mode="L")
                else:
                    # Try to take first 3 channels
                    return Image.fromarray(img_array[:, :, :3], mode="RGB")

        # If shape doesn't match expected formats
        raise TypeError(f"Unsupported NumPy array shape: {img.shape}")

    raise TypeError(
        "Input must be a PIL Image, NumPy array, or valid image path")


def apply_grayscale(img: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
    """
    Convert an image to grayscale.

    Args:
        img (PIL.Image, numpy.ndarray, or str): Input image in various formats

    Returns:
        numpy.ndarray: Grayscale version of the input image
    """
    # Ensure input is a PIL Image
    pil_img = ensure_pil_image(img)

    # Convert to grayscale
    grayscale_pil = pil_img.convert("L")

    # Convert back to numpy array
    grayscale_array = np.array(grayscale_pil)

    # If the input was a 3D array, maintain the 3D shape for consistency
    if isinstance(img, np.ndarray) and img.ndim == 3:
        return grayscale_array[:, :, np.newaxis]

    # Otherwise return the 2D array
    return grayscale_array


def apply_crop(img: Union[Image.Image, np.ndarray, str],
               corners: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop an image based on provided corner coordinates.

    Args:
        img (PIL.Image, numpy.ndarray, or str): Input image
        corners (tuple): (left, upper, right, lower) coordinates

    Returns:
        numpy.ndarray: Cropped image

    Raises:
        Exception: If coordinates are invalid
    """
    # Ensure input is a PIL Image
    pil_img = ensure_pil_image(img)

    # Check for negative values
    if any(val < 0 for val in corners):
        raise Exception("Coordinates cannot be negative")

    # Check coordinate bounds
    x_range, y_range = pil_img.size
    if (
        corners[0] > x_range
        or corners[2] > x_range
        or corners[1] > y_range
        or corners[3] > y_range
    ):
        raise Exception("Coordinates exceed image dimensions")

    # Crop and convert to numpy array
    return np.array(pil_img.crop(corners))


def apply_saturation(
    img: Union[Image.Image, np.ndarray, str], saturation_level: float
) -> np.ndarray:
    """
    Adjust image saturation.

    Args:
        img (PIL.Image, numpy.ndarray, or str): Input image
        saturation_level (float): Saturation multiplier
            0.0 = grayscale
            1.0 = original saturation
            > 1.0 = increased saturation

    Returns:
        numpy.ndarray: Saturated image

    Raises:
        ValueError: If saturation level is invalid
    """
    if saturation_level < 0:
        raise ValueError

    # Ensure input is a PIL Image
    pil_img = ensure_pil_image(img)

    # Validate saturation level
    if saturation_level < 0:
        raise ValueError("Saturation level cannot be negative")

    # Apply saturation
    enhancer = ImageEnhance.Color(pil_img)
    saturated_pil = enhancer.enhance(saturation_level)

    # Convert back to numpy array
    return np.array(saturated_pil)


def apply_edge_enhance(img: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
    """
    Enhance edges in the image.

    Args:
        img (PIL.Image, numpy.ndarray, or str): Input image

    Returns:
        numpy.ndarray: Edge-enhanced image
    """
    # Ensure input is a PIL Image
    pil_img = ensure_pil_image(img)

    # Apply edge enhancement
    enhanced_pil = pil_img.filter(ImageFilter.EDGE_ENHANCE)

    # Convert back to numpy array
    return np.array(enhanced_pil)


def apply_edge_detection(
        img: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
    """
    Detect edges in the image.

    Args:
        img (PIL.Image, numpy.ndarray, or str): Input image

    Returns:
        numpy.ndarray: Edge-detected image
    """
    # Ensure input is a PIL Image
    pil_img = ensure_pil_image(img)

    # Apply edge detection
    edge_pil = pil_img.filter(ImageFilter.FIND_EDGES)

    # Convert back to numpy array
    return np.array(edge_pil)


def sanitize_dimensional_image(pil_img: Image.Image) -> np.ndarray:
    """
    Sanitizes the dimensional aspects of the image
    """
    input_array = np.array(pil_img)

    # Convert to grayscale
    if input_array.ndim == 3:
        if input_array.shape[2] == 3 or input_array.shape[2] == 4:
            # Convert RGB/RGBA to grayscale
            gray_img = np.array(pil_img.convert("L"))
        else:
            # Handle other 3D arrays (like multi-channel images)
            gray_img = np.mean(input_array, axis=2)
    else:
        # Already grayscale
        gray_img = input_array

    return gray_img

def apply_cell_segmentation(img, model_path='./models/tissuenet_model_0019999.pth'):
    """
    Applies cell segmentation using CelloType.

    Args:
        img (numpy.ndarray): Input image array
        model_path (str): Path to the pre-trained model.

    Returns:
        numpy.ndarray: Segmentation mask
    """
    # Load model
    model = CelloTypePredictor(
        model_path=model_path,
        confidence_thresh=0.3,
        max_det=1000,
        device='cuda', 
        config_path='./configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml'
    )

    # Run segmentation
    mask = model.predict(img)
    return mask
