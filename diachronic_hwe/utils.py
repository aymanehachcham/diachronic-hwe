import os

from PIL import Image, ImageEnhance, ImageFilter
from typing import Optional
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def cutDoc(
        main_word: str,
        tokenizer: PreTrainedTokenizer,
        doc: Optional[str] = None,
        max_length: int = 120,
) -> Optional[str]:
    main_word = ' ' + main_word.strip() + ' '
    if doc is None:
        doc = main_word
        return doc

    tokens = tokenizer.tokenize(doc)
    main_token = tokenizer.tokenize(main_word)[0]

    try:
        main_index = tokens.index(main_token)
    except ValueError:
        return None

    start = max(0, main_index - max_length // 2)
    end = start + max_length

    if end > len(tokens):
        end = len(tokens)
        start = max(0, end - max_length)

    # Convert the tokens back to ids for decoding
    token_ids = tokenizer.convert_tokens_to_ids(tokens[start:end])

    # Decode the token ids back to a string
    sliced_doc = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return sliced_doc


def find_project_root(filename=None) -> str:
    """
    Find the root folder of the project.

    Args:
        filename (str): The name of the file to look for in the root folder.

    Returns:
        str: The path of the root folder.

    """
    # Get the path of the file that is being executed
    current_file_path = os.path.abspath(os.getcwd())

    # Navigate back until we either find a $filename file or there is no parent
    # directory left.
    root_folder = current_file_path
    while True:
        # Custom way to identify the project root folder
        if filename is not None:
            env_file_path = os.path.join(root_folder, filename)
            if os.path.isfile(env_file_path):
                break

        # Most common ways to identify a project root folder
        if os.path.isfile(os.path.join(root_folder, "pyproject.toml")) or os.path.isfile(
            os.path.join(root_folder, "config.toml")
        ):
            break

        parent_folder = os.path.dirname(root_folder)
        if parent_folder == root_folder:
            raise ValueError("Could not find the root folder of the project.")

        root_folder = parent_folder

    return root_folder


def find_closest(filename: str) -> str:
    """
    Find the closest file with the given name in the project root folder.

    Args:
        filename (str): The name of the file to look for in the root folder.

    Returns:
        str: The path of the file.
    """
    return os.path.join(find_project_root(filename), filename)


# function that converts jp2 to png keeping a high quality and resolution
def jp2_to_jpg(jp2_path: str, jpg_path: str) -> None:
    """
    Convert a jp2 file to jpg

    Args:
        jp2_path (str): The path to the jp2 file.
        jpg_path (str): The path to the jpg file.
    """
    img = Image.open(jp2_path)
    img.save(jpg_path, "JPEG", quality=100, subsampling=0)


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
    img = img.convert("1")
    img.save(img_path)


# Apply binarization to an image
def binarization(img_path: str) -> None:
    """
    Apply binarization to an image

    Args:
        img_path (str): The path to the image.
    """
    img = Image.open(img_path)
    img = img.convert("1")
    img.save(f"{img_path}-binarized.jpg")
