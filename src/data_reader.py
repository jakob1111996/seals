# This file implements the data reading functionality
import csv
from itertools import chain

import numpy as np
import pandas as pd

from faiss_searcher import get_image_uris_ordered

images_file = (
    "data/saved_metadata/oidv6-train-images-with-labels-with-rotation.csv"
)
classes_file = "data/saved_metadata/oidv6-class-descriptions.csv"
annotations_file = (
    "data/saved_metadata/oidv6-train-annotations-human-imagelabels.csv"
)


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
        self.eval_classes = None
        self.eval_class_ids = {}
        self.eval_class_indexes = {}
        self.choose_eval_classes()
        self.determine_indices_per_class()

    def get_positives_for_classes(self):
        """
        Determine the number of positive examples for each class in the dataset
        and the ids of images associated with each class.
        If an image contains the same class multiple times, it is only counted
        once. The human-created image labels are used.
        :return: Tuple of two dicts:
            0: Dictionary with class string as key and positive count as value
            1: Dictionary with class string as key and image id list as value
        """
        classes_list = self.get_csv_column(classes_file, 0)
        image_ids = {im_class: [] for im_class in classes_list}
        with open(annotations_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            _ = next(csv_reader)  # Skip first row
            for row in csv_reader:
                image_ids[row[2]].append(row[0])
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
        for class_name in self.eval_classes:
            self.eval_class_ids[class_name] = class_ids[class_name]

    def determine_indices_per_class(self):
        """
        This function determines the indices that the evaluated classes have
        in the Faiss IndexLSH object. This is required to find the embeddings
        for the seed set and to be able to quickly get labels.
        The results are stored in the eval_class_indexes dictionary in the
        instance of this class.
        We need to go through multiple files in the dataset files in order to
        get the FAISS indices for every class.
        """
        combined_ids = list(chain(*self.eval_class_ids.values()))
        all_ids = self.get_csv_column(images_file, 0)
        all_id_dict = dict(zip(all_ids, range(len(all_ids))))
        intermed_indices = [all_id_dict[element] for element in combined_ids]
        all_urls = self.get_csv_column(images_file, 2)
        needed_urls = [all_urls[index] for index in intermed_indices]
        all_urls_ordered = list(get_image_uris_ordered())
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
        The results are stored in the eval_class_indexes class member.
        :param all_indices: All indices for the evaluation classes
        """
        count = 0
        for class_name, ids in self.eval_class_ids:
            self.eval_class_indexes[class_name] = all_indices[
                count : count + len(ids)
            ]
            count += len(ids)
        assert count == len(all_indices)

    @staticmethod
    def get_csv_column(file_name: str, column: int = 0):
        """
        This function gets one column from an arbitrary csv file
        :param file_name: The name of the csv file
        :param column: The column index
        :return: List of the entries in the column
        """
        column_list = (
            pd.read_csv(file_name, sep=",", header=0, usecols=[column])
            .values.reshape((-1,))
            .tolist()
        )
        return column_list


if __name__ == "__main__":
    dr = DataManager()
