import os
import random
from typing import List, Dict
import itertools

import srsly

from .embeddings import RetrieveEmbeddings
from ..utils import cutDoc
from ..ai_finder.finder import GptSimilarityFinder
from tqdm import tqdm
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

class HWEPreTrain:
    """
    Class to pre-train the HWE model, curating and generating the hierarchical
    dataset.
    """

    def __init__(self, words_folder_path: str):
        if not os.path.exists(words_folder_path):
            raise FileNotFoundError(f"Folder {words_folder_path} does not exist.")
        if not os.path.isdir(words_folder_path):
            raise NotADirectoryError(f"{words_folder_path} is not a directory.")
        self.words_path_folder = words_folder_path
        self.embeddings = RetrieveEmbeddings()
        self.gpt = GptSimilarityFinder(os.getenv("GPT_HWE_2"))
        self.word_data = srsly.read_json(os.path.join(os.getenv("COMPILED_DOCS_PATH"), "senses.json"))

    def sample_per_file(self, file_path: str, target: str, samples: int = 30) -> List:
        """
        Sample a few documents per folder.
        """
        docs = []
        with open(file_path, "r") as f:
            lines = f.readlines()
        for line in tqdm(lines, desc=f"Sampling documents from file: {file_path}"):
            std_doc = self.embeddings.vector.standardize_word_variations(line, target)
            norm_doc = cutDoc(target, self.embeddings.vector.tokenizer, std_doc, max_length=150)
            docs += [norm_doc]

        if len(docs) < samples:
            return docs
        return random.sample(docs, samples)

    def prepare_samples(self, target: str) -> Dict:
        """
        Prepare the samples for the target word.
        """
        samples = {}
        for root, dirs, files in os.walk(self.words_path_folder):
            for directory in dirs:
                dir_name = directory.split(".")[0]
                if dir_name == target:
                    dir_path = os.path.join(root, directory)
                    # walk through the directory and get the files
                    for _, _, word_files in os.walk(dir_path):
                        for file in word_files:
                            if not file.endswith(".txt"):
                                continue
                            file_path = os.path.join(dir_path, file)
                            file_name = file.split(".")[0]
                            year = file_name.split("_")[1]
                            doc_samples = self.sample_per_file(file_path, target, samples=30)
                            samples[year] = doc_samples
        return samples

    def generate_ontology(self, target: str):
        """
        Call GPT-4 to generate the ontology for each word.
        """
        output = {}
        doc_samples: Dict = self.prepare_samples(target)
        target_senses: Dict = self.word_data[target].get("senses")
        target_sub_senses: List = self.word_data[target].get("sub_senses")
        for year, samples in doc_samples.items():
            logging.info(f"Processing year: {year} for word {target}...")
            for sample in samples:
                output.setdefault(int(year), []).append(self.gpt.call_gpt(
                    doc=sample,
                    target=target,
                    senses=target_senses,
                    sub_senses=target_sub_senses
                ))

        # transform output to a json file
        srsly.write_json(
            os.path.join(os.getenv("COMPILED_DOCS_PATH"), f"{target}_ontology.json"),
            output
        )

    @staticmethod
    def add_missing_relations(relations: List) -> List:
        """
        Creating the missing relations in terms of target and sub_senses:
        :param relations:
        :return:
        """
        # Dynamically extract the main sense from the first relation
        main_sense = relations[0].split(" - ")[0].strip()
        # Initial position to insert new relations right after the second row
        insert_position = 2

        # Create a set of all sub-senses directly linked to the main sense
        directly_linked_sub_senses = set()
        for relation in relations:
            sense, sub_sense = [item.strip() for item in relation.split(" - ")]
            if sense == main_sense:
                directly_linked_sub_senses.add(sub_sense)

        # Find sub-senses in the last five relations
        for relation in relations[-5:]:
            _, sub_sense = [item.strip() for item in relation.split(" - ")]

            # Check if this sub-sense is not already directly linked to the main sense
            if sub_sense not in directly_linked_sub_senses:
                # Create the new relation if it does not already exist
                new_relation = f"{main_sense} - {sub_sense}"
                if new_relation not in relations:
                    # Insert the new relation at the specified position
                    relations.insert(insert_position, new_relation)
                    # Update the set of directly linked sub-senses
                    directly_linked_sub_senses.add(sub_sense)
                    # Increment the insert position for any further new relations
                    insert_position += 1

        return relations

    def post_process_ontology(self, target: str):
        """
        Post process the ontology to get the final dataset.
        """
        ontology = srsly.read_json(
            os.path.join(os.getenv("COMPILED_DOCS_PATH"), f"{target}_ontology.json")
        )
        processed_relations = {}
        for year, relations in ontology.items():
            for relation in relations:
                relation.pop(0)
                relation.pop(-1)
                # strip each string in the relation:
                relation = [item.strip() for item in relation]
                relation = self.add_missing_relations(relation)
                processed_relations.setdefault(year, []).append(relation)

        srsly.write_json(
            os.path.join(os.getenv("COMPILED_DOCS_PATH"), f"{target}_processed_ontology.json"),
            processed_relations
        )

    def create_hwe_dataset(self, target_year: str):
        """
        Get the ontology for the target word per year.
        """
        ontology_per_year = {}
        targets = ['network', 'web', 'virus']
        for target in tqdm(targets, desc=f"Processing targets..."):
            ontology = srsly.read_json(
                os.path.join(os.getenv("COMPILED_DOCS_PATH"), f"{target}_processed_ontology.json")
            )
            for year, relations in ontology.items():
                if year == target_year:
                    ontology_per_year.setdefault(year, []).append(
                        list(itertools.chain(*relations))
                    )

        ontology_per_year[target_year] = list(itertools.chain(*ontology_per_year[target_year]))
        folder = os.getenv("TRAIN")
        # srsly.write_json(
        #     os.path.join(folder, f"{target_year}_hwe.json"),
        #     ontology_per_year
        # )
        # save the list as a tsv file also:
        with open(os.path.join(folder, f"{target_year}_hwe.tsv"), "w") as f:
            for relation in ontology_per_year[target_year]:
                tsv_line = relation.replace(" - ", "\t")
                f.write(f"{tsv_line}\n")



