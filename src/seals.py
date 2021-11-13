# This file implements the SEALS algorithm logic
from data_reader import DataManager

from faiss_searcher import FaissIndex


class SEALSAlgorithm:
    """
    This class implements the main logic behind the SEALS algorithm.
    It combines everything and runs the experiments
    """

    def __init__(self):
        self.faiss_index = FaissIndex()
        self.data_reader = DataManager(self.faiss_index)
