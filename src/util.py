# This file contains some utility functions required by several other files.
import csv
import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def read_parquet_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a parquet.gzip embedding file from the data folder
    :param file_path: The path of the file to read
    :return: Tuple of embeddings array (n, 256) and imageURI array (n,)
    """
    raw_data = pq.read_table(file_path)
    data = np.stack(np.array(raw_data[0].to_numpy()), axis=0)
    uris = np.stack(np.array(raw_data[1].to_numpy()), axis=0)
    return data, uris


def get_csv_column(file_name: str, column: int = 0) -> List:
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


def cleanup_annotations(
    data_folder: str,
    images_file: str,
    cleaned_annotations_file: str,
    annotations_file: str,
) -> set:
    """
    Problematically, not all images that are in the human image labels are
    still available on flickr and in the embeddings. Therefore, we need to
    clean the human image annotations and create a cleaned file once.
    """
    cleaned_uris = get_all_image_uris_ordered(data_folder)
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
            writer.writerow(["imageID", "class", "confidence"])
            for row in reader:
                if row[0] in id_set:
                    writer.writerow([row[0], row[2], row[3]])
    return uri_set


def get_all_image_uris_ordered(data_folder: str) -> np.ndarray:
    """
    Get all the image URIs ordered the same way as the corresponding
    embeddings in the saved_embeddings files. The returned URIs contain
    some URIs which are not labeled or don not exist in the metadata.
    :return: Array of image URIs
    """
    image_uris = np.empty((0,))
    for file in glob.glob(os.path.join(data_folder, "*.parquet.gzip")):
        data = pq.read_table(file)
        image_uris = np.concatenate(
            [image_uris, np.array(data[1].to_numpy())], axis=0
        )
    return image_uris
