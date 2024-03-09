import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from typing import Optional
# from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def cutDoc(
        main_word: str,
        tokenizer,
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


def percentages_to_floats(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = percentages_to_floats(value)
    elif isinstance(data, list):
        if all(isinstance(v, int) for v in data):
            return [v / 100.0 for v in data]
    return data



def fix_proportions(data, target):

    senses = data[target]["senses"]
    periods = 5
    for sense, sense_data in senses.items():
        proportion = sense_data["proportion"]
        subsenses = sense_data["subsenses"]

        for period in range(periods):
            p = proportion[period]
            if p == 0:
                number_of_children = 0
            
            else:
                for subsense, subsense_data in subsenses.items():
                    sub_p = subsense_data["proportion"][period]
                    if sub_p == 0:
                        subsense_data["proportion"][period] = 0
                    else:
                        subsense_data["proportion"][period] = sub_p / p

        



def get_attended(data, subsenses, period):
    data = data[str(period)]

    attended = {}
    for subsense in subsenses:
        attended[subsense] = []
        for d in data:
            if subsense in d:
                att = d.split(" - ")
                try:
                    att.remove(subsense)
                    attended[subsense].extend(att)
                
                except:
                    print(d, subsense, att)
                

        if len(attended[subsense]) > 0:
            attended[subsense] = list(set(attended[subsense]))
        
        else:
            attended.pop(subsense)
    
    return attended



def get_Dataset(data, target, period):
    senses: dict = data[target]["senses"]
    sense_list = list(senses.keys())

    pairs = {}
    periods = [1980,1990,2000,2010,2017]

    for p in range(len(periods)):
        if periods[p] != period:
            continue
        pairs[periods[p]] = []
        for sense in sense_list:
            sense_data: dict = senses[sense]
            subsenses: dict = sense_data["subsenses"]
            subsense_list = list(subsenses.keys())

            for subsense in subsense_list:
                subsense_data: dict = subsenses[subsense]
                children = subsense_data["children"][p]

                if children == 0:
                    continue

                for child in children:
                    s = sense
                    ss = subsense
                    c = child
                    t = target

                    leafs = [
                        {"child": s, "parent": t},
                        {"child": ss, "parent": s},
                        {"child": c, "parent": ss}
                    ]
                    # leafs = [(s,t), (ss,s), (c,ss)]
                    pairs[periods[p]].extend(leafs)
    
    df = pd.DataFrame(pairs[period], columns=["child", "parent"])
    return df



if __name__ == "__main__":
    import json

  
    target = "network"
    with open(f"input/network.json", "r") as f:
        data = json.load(f)
    
    period = 2017
    df = get_Dataset(data, target, period)

    print(df.head())

    print(df.shape)

    print(df.drop_duplicates().shape)
    df = df.drop_duplicates()

    df.to_csv(f"input/{target}_{period}.tsv", index=False, sep="\t", header=False)


   
