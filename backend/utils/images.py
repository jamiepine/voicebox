"""Image processing utilities for avatar uploads."""

from pathlib import Path
from typing import Optional, Tuple
from PIL import Image

# JPEG can be reported as 'JPEG' or 'MPO' (for multi-picture format from some cameras)
ALLOWED_FORMATS = {'PNG', 'JPEG', 'WEBP', 'MPO', 'JPG'}
MAX_SIZE = 512
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


def validate_image(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate image format and file size.
    
    Args:
        file_path: Path to image file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(file_path)
    
    # Check file size
    if path.stat().st_size > MAX_FILE_SIZE:
        return False, f"File size exceeds maximum of {MAX_FILE_SIZE // (1024 * 1024)}MB"
    
    try:
        with Image.open(file_path) as img:
            # Verify the image can be loaded
            img.load()
            
            # Check format (normalize JPEG variants)
            img_format = img.format
            if img_format in ('MPO', 'JPG'):
                img_format = 'JPEG'
            
            if img_format not in {'PNG', 'JPEG', 'WEBP'}:
                return False, f"Invalid format '{img_format}'. Allowed formats: PNG, JPEG, WEBP"
            
            return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def process_avatar(input_path: str, output_path: str, max_size: int = MAX_SIZE) -> None:
    """
    Process avatar image: resize and optimize.
    
    Resizes image to fit within max_size x max_size while maintaining aspect ratio.
    
    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        max_size: Maximum width or height in pixels
    """
    with Image.open(input_path) as img:
        # Handle EXIF orientation for JPEG images
        try:
            from PIL import ExifTags
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = img._getexif()
            if exif is not None:
                orientation_value = exif.get(orientation)
                if orientation_value == 3:
                    img = img.rotate(180, expand=True)
                elif orientation_value == 6:
                    img = img.rotate(270, expand=True)
                elif orientation_value == 8:
                    img = img.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError, TypeError):
            # No EXIF data or orientation tag
            pass
        
        # Convert to RGB if necessary (handles RGBA, P, CMYK, etc.)
        if img.mode not in ('RGB', 'L'):
            if img.mode == 'RGBA':
                # Create white background for RGBA images
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                img = background
            elif img.mode == 'CMYK':
                # Convert CMYK to RGB
                img = img.convert('RGB')
            elif img.mode == 'P':
                # Convert palette mode to RGB
                img = img.convert('RGB')
            else:
                img = img.convert('RGB')
        
        # Calculate new size maintaining aspect ratio
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Determine output format from extension
        output_ext = Path(output_path).suffix.lower()
        
        format_map = {
            '.png': 'PNG',
            '.jpeg': 'JPEG',
            '.jpg': 'JPEG',
            '.webp': 'WEBP'
        }
        
        output_format = format_map.get(output_ext, 'PNG')
        
        # Save with optimization
        save_kwargs = {'optimize': True}
        if output_format == 'JPEG':
            save_kwargs['quality'] = 90
        
        img.save(output_path, format=output_format, **save_kwargs)
