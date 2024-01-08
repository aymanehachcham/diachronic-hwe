

from hwe.data.extraction import NewsPaperExtractorXml
from hwe.data.processor import DocumentProcessor
import nltk
from nltk.corpus import wordnet as wn


def find_relations(word):
    synsets = wn.synsets(word)
    relations = []

    for synset in synsets:
        hypernyms = synset.hypernyms()
        hyponyms = synset.hyponyms()

    for hypernym in hypernyms:
        relations.append((synset.name(), hypernym.name(), 'hypernym'))

    for hyponym in hyponyms:
        relations.append((synset.name(), hyponym.name(), 'hyponym'))

    return relations

if __name__ == '__main__':

    word = 'gay'
    relations = find_relations(word)

    print(relations)
    


       





