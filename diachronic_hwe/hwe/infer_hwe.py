
import os
import srsly
from gensim.models.poincare import PoincareModel, PoincareKeyedVectors
import matplotlib.pyplot as plt


class InferPoincareEmbeddings:
    """
    Class to infer Poincare embeddings from a given model.
    """
    _model_path: str
    _vectors_path: str
    _model: PoincareModel
    _kv: PoincareKeyedVectors

    def __init__(self, model_dir: str):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Folder {model_dir} does not exist.")
        if not os.path.isdir(model_dir):
            raise NotADirectoryError(f"{model_dir} is not a directory.")
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith(".model"):
                    self._model_path = os.path.join(root, file)
                elif file.endswith(".w2vec"):
                    self._vectors_path = os.path.join(root, file)

        self.model = PoincareModel.load(self._model_path)

    def most_similar(self, word: str, topn: int = 10):
        """
        Get the most similar words to the given word.
        """
        similar = self.model.kv.most_similar(f'{word}.n.01', topn=topn)
        senses = srsly.read_json(os.path.join(os.getenv("COMPILED_DOCS_PATH"), "senses.json"))
        for idx, sim in enumerate(similar):
            target = sim[0].split(".")[0]
            if target == word or target == 'network':
                sense = senses[target].get("senses").get(sim[0])
                similar[idx] = (sense, sim[1])

        # return only the words from similar:
        return [sim[0] for sim in similar]

    def test_visualize(self):
        """
        Test the visualization of the embeddings.
        """
        pass

