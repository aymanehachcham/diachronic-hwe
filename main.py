import os

from diachronic_hwe.hwe.embeddings import RetrieveEmbeddings
from diachronic_hwe.hwe.sense_embeddings import VectorEmbeddings
from diachronic_hwe.hwe.preprocessing import NewsExamplesDataset
from diachronic_hwe.hwe.train import BertTrainer
from dotenv import load_dotenv
from diachronic_hwe.ai_finder.finder import GptSimilarityFinder
import srsly
from nltk.corpus import stopwords, wordnet as wn

# load_dotenv()

if __name__ == "__main__":
    # Example usage
    target_word = "abuse"  # Ensure this matches the tokenization
    input_text = "King, dressed in a three-piece gray flannel suit, quickly left the court building without talking to reporters. King, who played basketball at Fort Hamilton High School in Brooklyn and at the University of Tennessee, was suspended by the Jazz after his arrest last New Years Day. He was originally charged with three counts of forcible sodomy and two counts of forcible sexual abuse, all felonies. A 25-year-old Salt Lake City woman had charged that King had forced her to perform sexual acts after she had gone with him to his apartment. Still Faces Drug Charge But a plea-bargaining arrangement reached yesterday called for King to plead guilty to the two misdemeanor counts. The three charges of forcible sodomy are to be dropped, said John T. Nielsen, an assistant Salt Lake County attorney. W.C. Gwynn of the county attorneys office said he had no idea when a misdemeanor drug-possession charge against King would be heard in Circuit Court. The police who arrested King allegedly found cocaine in his apartment. Mr. Gwynn said that the woman involved in the case had no complaints about the agreement. At todays sentencing, the judge asked Kings attorney, Robert Van-Scyver, whether King was an alcoholic. Yes, Mr. VanScyver replied. Remains Under Suspension Mr. VanScyver said King had been attending meetings of Alcoholics Anonymous, and Judge Durham said King should continue to undertake treatment for alcoholism under the supervision of his parole officer. She told King that he could choose to remain out of trouble and continue his basketball career or end up spending a substantial part of his life in jail. King has had a long history of trouble with the law, primarily concerning drug possession, ever since he attended Tennessee. Officials of the Jazz had little comment after todays proceedings. The teams general manager, Frank Layden, said only, His status with us is unchanged, meaning that King remained suspended for now."  # noqa E501
    embeddings = RetrieveEmbeddings()
    context = embeddings.get_attended_words(input_text, target_word)
    target, pos = next(embeddings.pos_tag(input_text, target_word))
