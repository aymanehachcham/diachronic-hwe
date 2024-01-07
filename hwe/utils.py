from PIL import Image, ImageFilter, ImageEnhance

# function that converts jp2 to png keeping a high quality and resolution
def jp2_to_jpg(jp2_path: str, jpg_path: str) -> None:
    """
    Convert a jp2 file to jpg

    Args:
        jp2_path (str): The path to the jp2 file.
        jpg_path (str): The path to the jpg file.
    """
    img = Image.open(jp2_path)
    img.save(jpg_path, 'JPEG', quality=100, subsampling=0)


# Apply noise reduction to an image
def noise_reduction(img_path: str) -> None:
    """
    Apply noise reduction to an image

    Args:
        img_path (str): The path to the image.
    """
    img = Image.open(img_path)
    img = img.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    img = img.convert('1')
    img.save(img_path)

# Apply binarization to an image
def binarization(img_path: str) -> None:
    """
    Apply binarization to an image

    Args:
        img_path (str): The path to the image.
    """
    img = Image.open(img_path)
    img = img.convert('1')
    img.save(f'{img_path}-binarized.jpg')
