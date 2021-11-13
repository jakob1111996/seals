# This file implements the FAISS similarity search functionality
import csv
import glob
import os

import faiss
import numpy as np
import pyarrow.parquet as pq
from alive_progress import alive_bar

from definitions import cleaned_uri_file

data_folder = "data/saved_embeddings/train"


class FaissIndex:
    """
    This class wraps the Facebook AI Similarity Search framework
    which is used as a k-Nearest-Neighbors implementation.
    """

    def __init__(self, n_bits: int = 32, uri_set: set = None):
        """
        Constructor for the FaissIndex class
        :param n_bits: The number of bits used for the LSH hashes
        :param uri_set: Set of URIs that are allowed for the index creation
        """
        self.embedding_dim = 256
        self.n_bits = n_bits
        self.index = faiss.IndexLSH(self.embedding_dim, n_bits)
        self.index_file_name = "data/faiss.index"
        if os.path.exists(self.index_file_name):
            self.load_index()
        else:
            if not uri_set:
                raise RuntimeError(
                    "FaissIndex should have been created by the "
                    "first time setup run. Please re-run the "
                    "initial setup (see DataManager class)."
                )
            self.generate_index(uri_set)

    def add_data_to_index(self, data):
        """
        This method is used to add data to the LSH Index.
        :param data: The data to add to the index. Expecting shape (n, 256)
        """
        assert data.shape[1] == 256
        features = np.ascontiguousarray(np.asarray(data, "float32"))
        self.index.add(features)

    def generate_index(self, uri_set: set):
        """
        This method reads all the data and adds it to the IndexLSH object.
        Apart from that, it creates all other data files that are required
        by the SEALS algorithm later.
        """
        mm = np.memmap(
            "data/embeddings.bin",
            dtype="float32",
            mode="w+",
            shape=(len(uri_set), 256),
        )
        image_uris = np.empty((0,))
        line = 0
        with alive_bar(
            len(glob.glob(os.path.join(data_folder, "*.parquet.gzip"))),
            title="Generating index",
            force_tty=True,
        ) as bar:
            for file in glob.glob(os.path.join(data_folder, "*.parquet.gzip")):
                raw_data = pq.read_table(file)
                data = np.stack(np.array(raw_data[0].to_numpy()), axis=0)
                uris = np.stack(np.array(raw_data[1].to_numpy()), axis=0)
                drop_lines = []
                for index, uri in enumerate(uris):
                    if uri not in uri_set:
                        drop_lines.append(index)
                data = np.delete(data, drop_lines, 0)
                mm[line : line + data.shape[0], :] = data
                uris = np.delete(uris, drop_lines, 0)
                image_uris = np.concatenate([image_uris, uris], axis=0)
                mm.flush()
                line += data.shape[0]
                self.add_data_to_index(data)
                bar()
        with open(cleaned_uri_file, "w", newline="") as clean_file:
            writer = csv.writer(clean_file, delimiter=",")
            writer.writerow(["imageURI"])
            for uri in image_uris:
                writer.writerow([uri])
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


def get_all_image_uris_ordered() -> np.ndarray:
    """
    Get all the image URIs ordered the same way as the corresponding
    embeddings in the saved_embeddings files.
    :return: Array of image URIs
    """
    image_uris = np.empty((0,))
    for file in glob.glob(os.path.join(data_folder, "*.parquet.gzip")):
        data = pq.read_table(file)
        image_uris = np.concatenate(
            [image_uris, np.array(data[1].to_numpy())], axis=0
        )
    return image_uris


if __name__ == "__main__":
    fi = FaissIndex()
