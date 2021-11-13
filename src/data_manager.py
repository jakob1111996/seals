# This file implements the data reading functionality
import csv
import os
from itertools import chain
from typing import Iterable, Union

import numpy as np

from definitions import classes_file, cleaned_annotations_file, images_file
from faiss_searcher import FaissIndex, get_all_image_uris_ordered
from util import cleanup_annotations, get_csv_column


class DataManager:
    """
    The DataManager class is responsible for reading the dataset from the data
    folder. The data folder needs to be structured like this:
    data/
        saved_embeddings/
            train/
                some_name.parquet.gzip
            test/
                some_name.parquet.gzip
        saved_metadata/
            oidv6-class-descriptions.csv
            oidv6-train-annotations-human-imagelabels.csv
            oidv6-train-images-with-labels-with-rotation.csv
    """

    def __init__(self):
        """
        Constructor for the data reader class.
        :param faiss_index: The FaissIndex class
        """
        self.num_embeddings = 6_607_906
        self.eval_classes = None
        self.eval_class_ids = {}
        self.eval_class_indices = {}
        if not os.path.exists(cleaned_annotations_file):
            self.first_time_setup()
        self.choose_eval_classes()
        self.determine_faiss_indices_per_class()
        print("Data Setup completed successfully!")
        self.embedding_mm = np.memmap(
            "data/embeddings.bin",
            dtype="float32",
            mode="r",
            shape=(self.num_embeddings, 256),
        )

    @staticmethod
    def get_positives_for_classes():
        """
        Determine the number of positive examples for each class in the dataset
        and the ids of images associated with each class.
        If an image contains the same class multiple times, it is only counted
        once. The human-created image labels are used.
        :return: Tuple of two dicts:
            0: Dictionary with class string as key and positive count as value
            1: Dictionary with class string as key and image id list as value
        """
        classes_list = get_csv_column(classes_file, 0)
        image_ids = {im_class: [] for im_class in classes_list}
        with open(cleaned_annotations_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            _ = next(csv_reader)  # Skip first row
            for row in csv_reader:
                if row[2] == "1":
                    image_ids[row[1]].append(row[0])
        class_counts = {}
        class_ids = {}
        for key, value in image_ids.items():
            class_counts[key] = len(np.unique(value))
            class_ids[key] = np.unique(value)
        return class_counts, class_ids

    def choose_eval_classes(self):
        """
        This method chooses 200 classes with between 100 to 6,817 positive
        training examples which are then used for the SEALS algorithm.
        The classes are then stored in the data reader instance.
        """
        class_counts, class_ids = self.get_positives_for_classes()
        possible_classes = []
        for im_class, count in class_counts.items():
            if 6818 > count > 99:
                possible_classes.append(im_class)
        self.eval_classes = np.random.choice(possible_classes, 200, False)
        print("Selected 200 random classes for evaluation.")
        for class_name in self.eval_classes:
            self.eval_class_ids[class_name] = class_ids[class_name]

    def determine_faiss_indices_per_class(self):
        """
        This function determines the indices that the evaluated classes have
        in the Faiss IndexLSH object. This is required to find the embeddings
        for the seed set and to be able to quickly get labels.
        The results are stored in the eval_class_indices dictionary in the
        instance of this class.
        We need to go through multiple files in the dataset files in order to
        get the FAISS indices for every class.
        """
        print("Preparing data for chosen classes.")
        combined_ids = list(chain(*self.eval_class_ids.values()))
        all_ids = get_csv_column(images_file, 0)
        all_id_dict = dict(zip(all_ids, range(len(all_ids))))
        intermed_indices = [all_id_dict[element] for element in combined_ids]
        all_urls = get_csv_column(images_file, 2)
        needed_urls = [all_urls[index] for index in intermed_indices]
        all_urls_ordered = list(get_all_image_uris_ordered())
        all_url_dict = dict(
            zip(all_urls_ordered, range(len(all_urls_ordered)))
        )
        combined_class_inds = [
            all_url_dict[element] for element in needed_urls
        ]
        self.generate_index_dictionary(combined_class_inds)

    def generate_index_dictionary(self, all_indices):
        """
        Split all the indices that are determined by the
        determine_indices_per_class method into the separate classes
        analogous to the eval_class_ids dictionary.
        The results are stored in the eval_class_indices class member.
        :param all_indices: All indices for the evaluation classes
        """
        count = 0
        for class_name, ids in self.eval_class_ids.items():
            self.eval_class_indices[class_name] = all_indices[
                count : count + len(ids)
            ]
            count += len(ids)
        assert count == len(all_indices)

    @staticmethod
    def first_time_setup():
        """
        This function is only executed once when first running the code.
        It sets up all the required files and creates the faiss index.
        :return:
        """
        print("First time setup required. Starting.")
        uri_set = cleanup_annotations()
        _ = FaissIndex(uri_set=uri_set)
        print("First time setup finished.")

    def get_embedding(self, index: Union[int, Iterable[int]]) -> np.ndarray:
        """
        Get one embedding from the embedding memmap.
        :param index: The index or indices of the embedding in the faiss index.
        :return: ndarray of shape (n, 256) containing the embedding
        """
        return self.embedding_mm[index, :]


if __name__ == "__main__":
    dr = DataManager()
