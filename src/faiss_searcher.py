import glob
import os

import faiss
import numpy as np
import pyarrow.parquet as pq
from alive_progress import alive_bar


class FaissIndex:
    """
    This class wraps the Facebook AI Similarity Search framework
    which is used as a k-Nearest-Neighbors implementation.
    """

    def __init__(self, n_bits: int = 32):
        """
        Constructor for the FaissIndex class
        :param n_bits: The number of bits used for the LSH hashes
        """
        self.embedding_dim = 256
        self.n_bits = n_bits
        self.index = faiss.IndexLSH(self.embedding_dim, n_bits)
        self.index_file_name = "data/faiss.index"
        if os.path.exists(self.index_file_name):
            self.load_index()
        else:
            self.generate_index()

    def add_data_to_index(self, data):
        """
        This method is used to add data to the LSH Index.
        :param data: The data to add to the index. Expecting shape (n, 256)
        """
        assert data.shape[1] == 256
        features = np.ascontiguousarray(np.asarray(data, "float32"))
        self.index.add(features)

    def generate_index(self):
        """
        This method reads all the data and adds it to the IndexLSH object.
        """
        data_folder = "data/saved_embeddings/train"
        with alive_bar(
            len(glob.glob(os.path.join(data_folder, "*.parquet.gzip"))),
            title="Generating index",
            force_tty=True,
        ) as bar:
            for file in glob.glob(os.path.join(data_folder, "*.parquet.gzip")):
                data = pq.read_table(file)
                self.add_data_to_index(
                    np.stack(np.array(data[0].to_numpy()), axis=0)
                )
                bar()
        self.save_index()

    def save_index(self, file_name: str = "data/faiss.index"):
        """
        Saves the generated index in a file
        :param file_name: The name of the index file
        """
        faiss.write_index(self.index, file_name)

    def load_index(self, file_name: str = "data/faiss.index"):
        """
        Loads the index from an index file
        :param file_name: The name of the index file
        """
        self.index = faiss.read_index(file_name)


if __name__ == "__main__":
    fi = FaissIndex()
