# This file implements the FAISS similarity search functionality
import csv
import glob
import os
from typing import List

import faiss
import numpy as np
from alive_progress import alive_bar

from src.definitions import cleaned_train_uri_file
from src.util import read_parquet_file


class FaissIndex:
    """
    This class wraps the Facebook AI Similarity Search framework
    which is used as a k-Nearest-Neighbors implementation.
    """

    def __init__(self, n_bits: int = 24, uri_set: set = None):
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
            self.read_embeddings(
                uri_set, "data/train_embeddings.bin", cleaned_train_uri_file
            )

    def add_data_to_index(self, data: np.ndarray) -> None:
        """
        This method is used to add data to the LSH Index.
        :param data: The data to add to the index. Expecting shape (n, 256)
        """
        assert data.shape[1] == 256
        features = np.ascontiguousarray(np.asarray(data, "float32"))
        self.index.add(features)

    def search(self, data: np.ndarray, k: int = 100) -> np.ndarray:
        """
        Search through the index and return the k nearest neighbors of the
        search data in the index.
        :param data: The data to search for in shape (n, 256)
        :param k: The number of nearest neighbors to return for each sample
        :return: Array of indices of the nearest neighbors, shape (n*k,)
        """
        features = np.ascontiguousarray(np.asarray(data, "float32"))
        _, indices = self.index.search(features, k)
        return indices.reshape((-1,))

    def read_embeddings(
        self,
        uri_set: set,
        embeddings_file: str,
        write_uri_file: str,
        data_folder: str = "data/saved_embeddings/train",
        create_index: bool = True,
    ) -> None:
        """
        This method reads all the data and adds it to the IndexLSH object.
        Apart from that, it creates some other data files that are required
        by the SEALS algorithm later. It is all done in one function in order
        to not go through the large data files multiple times.
        :param uri_set: The set of URIs that are allowed in the
        :param embeddings_file: File for the embeddings memmap
        :param write_uri_file: File to store all the URIs in
        :param data_folder: The folder to read the dataset from
        :param create_index: Flag that determines whether the files are added
        to the faiss IndexLSH object or not.
        """
        mm = np.memmap(
            embeddings_file,
            dtype="float32",
            mode="w+",
            shape=(len(uri_set), 256),
        )
        image_uris = np.empty((0,))
        line = 0
        with alive_bar(
            len(glob.glob(os.path.join(data_folder, "*.parquet.gzip"))),
            title="Reading embeddings",
            force_tty=True,
        ) as bar:
            for file in glob.glob(os.path.join(data_folder, "*.parquet.gzip")):
                data, uris = read_parquet_file(file)
                drop_lines = self.find_unlabeled_data(uris, uri_set)
                # Save required embeddings in memmap
                data = np.delete(data, drop_lines, 0)
                mm[line : line + data.shape[0], :] = data
                mm.flush()
                # Store the imageURIs corresponding to the embeddings
                uris = np.delete(uris, drop_lines, 0)
                image_uris = np.concatenate([image_uris, uris], axis=0)
                line += data.shape[0]
                if create_index:
                    self.add_data_to_index(data)
                bar()

        self.save_uris(write_uri_file, image_uris)
        if create_index:
            self.save_index()  # Save faiss index to a file for later use

    @staticmethod
    def find_unlabeled_data(uris: np.ndarray, uri_set: set) -> List:
        """
        Find data that has no labels, and thus can not be used.
        :param uris: All URIs read from the embedding file.
        :param uri_set: Set of all allowed URIs
        :return:
        """
        drop_lines = []
        for index, uri in enumerate(uris):
            if uri not in uri_set:
                drop_lines.append(index)
        return drop_lines

    @staticmethod
    def save_uris(write_uri_file: str, uris: np.ndarray) -> None:
        """
        Save the URIs in uris in a csv file
        :param write_uri_file: The file path of the csv file to be created
        :param uris: All URIs to store in the file in an array
        """
        with open(write_uri_file, "w", newline="") as clean_file:
            writer = csv.writer(clean_file, delimiter=",")
            writer.writerow(["imageURI"])
            for uri in uris:
                writer.writerow([uri])

    def save_index(self, file_name: str = "data/faiss.index") -> None:
        """
        Saves the generated index in a file
        :param file_name: The name of the index file
        """
        faiss.write_index(self.index, file_name)

    def load_index(self, file_name: str = "data/faiss.index") -> None:
        """
        Loads the index from an index file
        :param file_name: The name of the index file
        """
        self.index = faiss.read_index(file_name)


if __name__ == "__main__":
    fi = FaissIndex()
