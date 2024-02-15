from diachronic_hwe.data.processor import DocumentProcessor
from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.test.utils import datapath
from nltk.corpus import wordnet as wn
import spacy

nlp = spacy.load("en_core_web_md")


def extract_paragraphs_with_all_forms(text, target_word):
    """
    Extracts paragraphs from text containing all forms of the target word using spaCy for lemmatization.

    :param text: The input text from which to extract paragraphs.
    :param target_word: The root form of the word to search for.
    :return: A list of paragraphs containing any form of the target word.
    """
    # Lemmatize the target word to get its base form
    target_lemma = nlp(target_word)[0].lemma_.lower()
    print(target_lemma)
    # Split the text into paragraphs
    paragraphs = text.split('\n')

    # Initialize a list to hold paragraphs that contain any form of the target word
    matching_paragraphs = []

    # Process each paragraph with spaCy
    for paragraph in paragraphs:
        if paragraph.strip():  # Ensure paragraph is not just whitespace
            doc = nlp(paragraph)
            # Check if any word in the paragraph has the same lemma as the target word
            for word in doc:
                if word.lemma_.lower() == target_lemma:
                    print(target_lemma)
                    matching_paragraphs.append(paragraph)

    return "\n".join(matching_paragraphs)


if __name__ == "__main__":
    # This is the main entry point
    # docs = DocumentProcessor(json_file_path="sample_data/TheNewYorkTimes1980.xml").retrieve_context(
    #     target_word="abuse",
    #     query="The law prohibits abuse in many forms."
    # )
    # for doc in docs:
    #     print(doc)
    #     print("\n\n")
    docs = DocumentProcessor(
        json_file_path="sample_data/TheNewYorkTimes1980.xml"
    ).retrieve_context_docs(target_word="abuse")
    for doc in docs:
        print(doc)
        print("\n\n")