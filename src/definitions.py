# The images file contains a list of all images with IDs and URIs
images_file = (
    "data/saved_metadata/oidv6-train-images-with-labels-with-rotation.csv"
)
# The classes file contains all available classes and descriptions
classes_file = "data/saved_metadata/oidv6-class-descriptions.csv"
# The annotations file contains all human-made annotations and labels for IDs
annotations_file = (
    "data/saved_metadata/oidv6-train-annotations-human-imagelabels.csv"
)
# This file contains a subset of the annotations that is shared between the
# embeddings and the images file
cleaned_annotations_file = "data/cleaned-annotations-human-imagelabels.csv"
# This file contains a list of all uris in the order of the faiss index
cleaned_uri_file = "data/cleaned-uris.csv"
