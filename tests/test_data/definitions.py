# This file contains some constants like file paths that do not change

# The images file contains a list of all images with IDs and URIs
train_images_file = "tests/test_data/metadata/train_images.csv"
# The classes file contains all available classes and descriptions
classes_file = "tests/test_data/metadata/oidv6-class-descriptions.csv"
# The annotations file contains all human-made annotations and labels for IDs
train_annotations_file = "tests/test_data/metadata/train_labels.csv"
# This file contains a subset of the annotations that is shared between the
# embeddings and the images file
cleaned_train_annotations_file = (
    "tests/test_data/metadata/cleaned-train-annotations.csv"
)
# This file contains a list of all uris in the order of the faiss index
cleaned_train_uri_file = "tests/test_data/metadata/cleaned-train-uris.csv"
cleaned_test_uri_file = "tests/test_data/metadata/cleaned-test-uris.csv"
# This file contains all test set annotations
test_annotations_file = "tests/test_data/metadata/test_labels.csv"
# This file contains a list of all test images with IDs and URIs
test_images_file = "tests/test_data/metadata/test_images.csv"
# This file contains a subset of the annotations that is shared between the
# test embeddings and the test images file
cleaned_test_annotations_file = (
    "tests/test_data/metadata/cleaned-test-annotations.csv"
)
# Training data folder
train_folder = "tests/test_data/train"
# Test data folder
test_folder = "tests/test_data/test"
# Training embeddings cleaned
train_embeddings_file = "tests/test_data/train_embeddings.bin"
# Testing embeddings cleaned
test_embeddings_file = "tests/test_data/test_embeddings.bin"
# Classes from the paper
used_classes_file = "tests/test_data/metadata/used_classes.csv"
# Faiss index file
faiss_index_file = "tests/test_data/faiss.index"
