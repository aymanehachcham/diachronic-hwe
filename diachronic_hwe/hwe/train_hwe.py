import toml
from gensim.models.poincare import PoincareModel, PoincareRelations, PoincareKeyedVectors
import os

from ..utils import find_closest


class PoincareEmbeddings:
    """
    Class to train Poincare Embeddings
    """

    def __init__(self, data_path: str):
        # Validate data path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Folder {data_path} does not exist.")
        if not data_path.endswith(".tsv"):
            raise ValueError("The data path should be a TSV file.")

        self.data_path_name = os.path.basename(data_path)
        self.year = self.data_path_name.split("_")[0]

        # Load configuration and prepare model configuration
        config_path = find_closest("diachronic_hwe/hwe/config/config.toml")
        config = toml.load(config_path)
        model_cfg = config["poincare"]

        # Initialize relations and update model configuration with training data
        relations = PoincareRelations(file_path=data_path, delimiter="\t")
        model_cfg["train_data"] = relations  # Directly assign relations to model_cfg

        # Initialize PoincareModel
        self.model = PoincareModel(**model_cfg)
        self.train_cfg = config["train"]
        self.model_cfg = model_cfg

    def train(self):
        """
        Train the Poincare model
        """
        self.model.train(**self.train_cfg)
        root_folder = os.getenv("HWE_FOLDER")
        packages = os.path.join(root_folder, "packages")
        if not os.path.exists(packages):
            os.makedirs(packages)

        model_name = f"{self.year}_poincare_gensim"
        model_path = os.path.join(packages, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # save the model:
        self.model.save(os.path.join(
            model_path,
            f"{model_name}_epochs_{self.train_cfg['epochs']}_neg_{self.model_cfg['negative']}"
        ))
        self.model.kv.save(os.path.join(
            model_path,
            f"{model_name}_epochs_{self.train_cfg['epochs']}_neg_{self.model_cfg['negative']}_vectors"
        ))

    def load(self):
        """
        Load the custom model
        """
        root_folder = os.path.join(os.getenv("HWE_FOLDER"), "packages")
        model = PoincareModel.load(os.path.join(
            root_folder,
            f"{self.year}_poincare_gensim",
            f"{self.year}_poincare_gensim_epochs_{self.train_cfg['epochs']}_neg_{self.model_cfg['negative']}"
        ))
        return model


