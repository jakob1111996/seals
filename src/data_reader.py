import csv

import numpy as np


class DataReader:
    """
    The DataReader class is responsible for reading the dataset from the data
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
        """
        self.eval_classes = None
        self.eval_class_ids = {}
        self.determine_eval_classes()

    @staticmethod
    def get_all_image_classes():
        """
        Return a list of all classes that are available in the dataset.
        The classes are represented by a string that looks like '/m/0dgw9r'
        :return: List of class strings
        """
        classes = []
        classes_file = "data/saved_metadata/" "oidv6-class-descriptions.csv"
        with open(classes_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            _ = next(csv_reader)  # Skip first row
            for row in csv_reader:
                classes.append(row[0])
        return classes

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
        image_ids = {im_class: [] for im_class in self.get_all_image_classes()}
        annotations_file = (
            "data/saved_metadata/oidv6-train-annotations-human-imagelabels.csv"
        )
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

    def determine_eval_classes(self):
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


if __name__ == "__main__":
    dr = DataReader()
