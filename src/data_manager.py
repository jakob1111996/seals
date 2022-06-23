# This file implements the data reading functionality
import csv
import os
from itertools import chain
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np

from src.definitions import *  # noqa: F403
from src.faiss_searcher import FaissIndex
from src.util import cleanup_annotations, get_csv_column


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

    def __init__(
        self,
        num_classes: int = 10,
        random_classes: bool = False,
        num_embeddings: int = 6_607_906,
        num_test_embeddings: int = 113_508,
    ) -> None:
        """
        Constructor for the data reader class.
        :param num_classes: The number of evaluation classes
        :param random_classes: Classes can either be selected randomly from
            all classes in the test set with (100, 6817) elements or we can
            use the classes used in the SEALS paper
        """
        self.num_embeddings = num_embeddings
        self.num_test_embeddings = num_test_embeddings
        self.num_classes = num_classes
        self.eval_classes = None
        self.eval_class_ids = {}
        self.eval_class_indices = {}
        if not os.path.exists(cleaned_train_annotations_file):
            self.first_time_setup()
        self.choose_eval_classes(num_classes, random=random_classes)
        self.get_indices_for_positives()
        print("Data Setup completed successfully!")
        self.embedding_mm = np.memmap(
            train_embeddings_file,
            dtype="float32",
            mode="r",
            shape=(self.num_embeddings, 256),
        )

    @staticmethod
    def get_positives_for_classes(annotations_file: str) -> Tuple[Dict, Dict]:
        """
        Determine the number of positive examples for each class in the dataset
        and the ids of images associated with each class.
        If an image contains the same class multiple times, it is only counted
        once. The human-created image labels are used.
        :param annotations_file: The file containing cleaned image annotations
        :return: Tuple of two dicts:
            0: Dictionary with class string as key and positive count as value
            1: Dictionary with class string as key and image id list as value
        """
        classes_list = get_csv_column(classes_file, 0)
        image_ids = {im_class: [] for im_class in classes_list}
        with open(annotations_file) as csv_file:
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

    def choose_eval_classes(
        self, class_count: int = 10, random: bool = False
    ) -> None:
        """
        This method chooses 200 classes with between 100 to 6,817 positive
        training examples which are then used for the SEALS algorithm.
        The classes are then stored in the data reader instance.
        :param class_count: The number of classes that shall be evaluated
        :param random: Select random classes for evaluation or use the
            predefined ones from the SEALS paper
        """
        print(f"Selecting {class_count} random classes for evaluation.")
        class_counts, class_ids = self.get_positives_for_classes(
            cleaned_train_annotations_file
        )
        possible_classes = []
        test_counts, test_ids = self.get_positives_for_classes(
            cleaned_test_annotations_file
        )
        if random:
            for im_class, count in class_counts.items():
                # We only consider classes with between 100 and 6817 training
                # examples and at least 50 positive test samples for evaluation
                if 6818 > count > 99 and test_counts[im_class] > 10:
                    possible_classes.append(im_class)
            self.eval_classes = np.random.choice(
                possible_classes, class_count, False
            )
        else:
            self.eval_classes = np.random.choice(
                self.read_classes_from_file(), class_count, False
            )

        for class_name in self.eval_classes:
            self.eval_class_ids[class_name] = class_ids[class_name]

    def get_indices_for_positives(self) -> None:
        """
        This function determines the indices that the evaluated classes have
        in the Faiss IndexLSH object. This is required to find the embeddings
        for the seed set and to be able to quickly get labels.
        The results are stored in the eval_class_indices dictionary in the
        instance of this class.
        """
        print("Preparing data for chosen classes.")
        combined_ids = list(chain(*self.eval_class_ids.values()))
        combined_indices = self.get_indices_from_ids(
            combined_ids, train_images_file, cleaned_train_uri_file
        )
        self.generate_index_dictionary(combined_indices)

    @staticmethod
    def get_indices_from_ids(
        ids: List, images_file: str, uri_file: str
    ) -> List[int]:
        """
        This function takes some imageIDs as input and returns their index in
        the faiss index and the embeddings instead.
        :param ids: The ids we want to map to indexes in a list
        :param images_file: The file that contains all images
        :param uri_file: The file that contains the cleaned uris
        :return: List of indices
        """
        all_ids = get_csv_column(images_file, 0)
        all_id_dict = dict(zip(all_ids, range(len(all_ids))))
        intermed_indices = [all_id_dict[element] for element in ids]
        all_urls = get_csv_column(images_file, 2)
        needed_urls = [all_urls[index] for index in intermed_indices]
        urls_ordered = get_csv_column(uri_file, 0)
        url_dict = dict(zip(urls_ordered, range(len(urls_ordered))))
        return [url_dict[element] for element in needed_urls]

    def generate_index_dictionary(self, all_indices: List) -> None:
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
    def first_time_setup() -> None:
        """
        This function is only executed once when first running the code.
        It sets up all the required files and creates the faiss index.
        """
        print("First time setup required. Starting.")
        train_uri_set = cleanup_annotations(
            train_folder,
            train_images_file,
            cleaned_train_annotations_file,
            train_annotations_file,
        )
        fi = FaissIndex(uri_set=train_uri_set, index_file=faiss_index_file)
        test_uri_set = cleanup_annotations(
            test_folder,
            test_images_file,
            cleaned_test_annotations_file,
            test_annotations_file,
        )
        fi.read_embeddings(
            test_uri_set,
            test_embeddings_file,
            cleaned_test_uri_file,
            test_folder,
            False,
        )
        print("First time setup finished.")

    def get_embedding(self, index: Union[int, Iterable[int]]) -> np.ndarray:
        """
        Get one embedding from the embedding memmap.
        :param index: The index or indices of the embedding in the faiss index.
        :return: ndarray of shape (n, 256) containing the embedding
        """
        return self.embedding_mm[index, :]

    def get_test_data(self, eval_class: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function returns the test data for the given class
        :param eval_class: The class we are evaluating currently
        :return: Tuple of embeddings and labels from the test set:
            0: embeddings: np.ndarray in format (n, 256)
            1: labels: np.ndarray in format (n,) with entries 0 or 1
        """
        embeddings = np.array(
            np.memmap(
                test_embeddings_file,
                dtype="float32",
                mode="r",
                shape=(self.num_test_embeddings, 256),
            )
        )
        counts, ids = self.get_positives_for_classes(
            cleaned_test_annotations_file
        )
        positive_ids = ids[eval_class]
        positive_indices = self.get_indices_from_ids(
            positive_ids, test_images_file, cleaned_test_uri_file
        )
        labels = np.zeros((embeddings.shape[0],))
        labels[positive_indices] = 1
        return embeddings, labels

    @staticmethod
    def read_classes_from_file(file_path: str = used_classes_file):
        """
        Read the classes used in SEALS from the provided csv file.
        :param file_path: The path to the csv file with specified classes.
        :return: The IDs of the classes as a list
        """
        class_names = get_csv_column(file_path, 0)
        class_dict = {}
        with open(classes_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            _ = next(csv_reader)  # Skip first row
            for row in csv_reader:
                class_dict[row[1]] = row[0]
        class_ids = [class_dict[class_name] for class_name in class_names]
        return class_ids


if __name__ == "__main__":
    dr = DataManager()
