import csv

import pandas as pd

from src.definitions import (
    annotations_file,
    cleaned_annotations_file,
    images_file,
)
from src.faiss_searcher import get_all_image_uris_ordered


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


def cleanup_annotations():
    """
    Problematically, not all images that are in the human image labels are
    still available on flickr and in the embeddings. Therefore, we need to
    clean the human image annotations and create a cleaned file once.
    """
    cleaned_uris = get_all_image_uris_ordered()
    all_uris = get_csv_column(images_file, 2)
    print(f"Total image count: {len(all_uris)}")
    print(f"Embeddings  count: {len(cleaned_uris)}")
    all_ids = get_csv_column(images_file, 0)
    uri_id_map = dict(zip(all_uris, all_ids))
    del all_uris
    del all_ids
    id_set = set()
    uri_set = set()
    for uri in cleaned_uris:
        if uri in uri_id_map:
            id_set.add(uri_id_map[uri])
            uri_set.add(uri)
    print(f"Common image count: {len(id_set)}")
    del uri_id_map
    del cleaned_uris

    with open(cleaned_annotations_file, "w", newline="") as clean_file:
        with open(annotations_file) as raw_file:
            reader = csv.reader(raw_file, delimiter=",")
            writer = csv.writer(clean_file, delimiter=",")
            _ = next(reader)  # Skip first row
            for row in reader:
                if row[0] in id_set:
                    writer.writerow([row[0], row[2], row[3]])
    return uri_set
