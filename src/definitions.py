# This file contains some constants like file paths that do not change

# The images file contains a list of all images with IDs and URIs
train_images_file = (
    "data/saved_metadata/oidv6-train-images-with-labels-with-rotation.csv"
)
# The classes file contains all available classes and descriptions
classes_file = "data/saved_metadata/oidv6-class-descriptions.csv"
# The annotations file contains all human-made annotations and labels for IDs
train_annotations_file = (
    "data/saved_metadata/oidv6-train-annotations-human-imagelabels.csv"
)
# This file contains a subset of the annotations that is shared between the
# embeddings and the images file
cleaned_train_annotations_file = "data/cleaned-train-annotations.csv"
# This file contains a list of all uris in the order of the faiss index
cleaned_train_uri_file = "data/cleaned-train-uris.csv"
cleaned_test_uri_file = "data/cleaned-test-uris.csv"
# This file contains all test set annotations
test_annotations_file = (
    "data/saved_metadata/test-annotations-human-imagelabels.csv"
)
# This file contains a list of all test images with IDs and URIs
test_images_file = "data/saved_metadata/test-images-with-rotation.csv"
# This file contains a subset of the annotations that is shared between the
# test embeddings and the test images file
cleaned_test_annotations_file = "data/cleaned-test-annotations.csv"
# Training data folder
train_folder = "data/saved_embeddings/train"
# Test data folder
test_folder = "data/saved_embeddings/test"
