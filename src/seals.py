from data_reader import DataReader


class SEALSAlgorithm:
    """
    This class implements the main logic behind the SEALS algorithm.
    It combines everything and runs the experiments
    """

    def __init__(self):
        self.data_reader = DataReader()
        self.eval_classes = self.data_reader.get_eval_classes()
