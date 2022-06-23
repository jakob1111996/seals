# This file tests the first time setup of the data files
import os.path
import sys

sys.modules["src.definitions"] = __import__(
    "tests.test_data.definitions",
    globals(),
    locals(),
    ["cleaned_train_uri_file"],
    0,
)

from src.data_manager import DataManager  # noqa: E402
from src.util import get_all_image_uris_ordered, get_csv_column  # noqa: E402


def test_initial_data_setup():
    if os.path.exists(
        "tests/test_data/metadata/cleaned-train-annotations.csv"
    ):
        os.remove("tests/test_data/metadata/cleaned-train-annotations.csv")
    if os.path.exists("tests/test_data/faiss.index"):
        os.remove("tests/test_data/faiss.index")
    dm = DataManager(1, False, 4, 4)
    assert os.path.exists("tests/test_data/train_embeddings.bin")
    assert os.path.exists("tests/test_data/test_embeddings.bin")
    assert os.path.exists(
        "tests/test_data/metadata/cleaned-test-annotations.csv"
    )
    assert (
        len(
            get_csv_column(
                "tests/test_data/metadata/cleaned-test-annotations.csv", 0
            )
        )
        == 25
    )
    assert os.path.exists(
        "tests/test_data/metadata/cleaned-train-annotations.csv"
    )
    assert (
        len(
            get_csv_column(
                "tests/test_data/metadata/cleaned-train-annotations.csv", 0
            )
        )
        == 25
    )
    assert os.path.exists("tests/test_data/metadata/cleaned-train-uris.csv")
    assert (
        len(
            get_csv_column(
                "tests/test_data/metadata/cleaned-train-uris.csv", 0
            )
        )
        == 4
    )
    assert os.path.exists("tests/test_data/metadata/cleaned-test-uris.csv")
    assert (
        len(
            get_csv_column("tests/test_data/metadata/cleaned-test-uris.csv", 0)
        )
        == 4
    )
    assert dm.num_embeddings == 4
    assert dm.num_test_embeddings == 4
    assert dm.num_classes == 1
    assert len(dm.eval_classes) == 1
    assert dm.eval_classes[0] in get_csv_column(
        "tests/test_data/metadata/train_labels.csv", 2
    )
    # Cleanup
    os.remove("tests/test_data/train_embeddings.bin")
    os.remove("tests/test_data/test_embeddings.bin")
    os.remove("tests/test_data/metadata/cleaned-test-annotations.csv")
    os.remove("tests/test_data/metadata/cleaned-train-annotations.csv")
    os.remove("tests/test_data/metadata/cleaned-train-uris.csv")
    os.remove("tests/test_data/metadata/cleaned-test-uris.csv")
    os.remove("tests/test_data/faiss.index")


def test_read_all_uris():
    uris = [
        "https://c1.staticflickr.com/1/1/1018404_0e4be5dd4b_o.jpg",
        "https://c1.staticflickr.com/1/1/1028428_02ced027d2_o.jpg",
        "https://c1.staticflickr.com/1/1/1032097_f3ec6d5925_o.jpg",
        "https://c1.staticflickr.com/1/1/1037372_9f061ca8ae_o.jpg",
        "https://c1.staticflickr.com/1/1/1049564_e9f5becef6_o.jpg",
        "https://c1.staticflickr.com/1/1/1049626_26e9df4089_o.jpg",
        "https://c1.staticflickr.com/1/1/1049840_11ba694f34_o.jpg",
        "https://c1.staticflickr.com/1/1/1053074_64a000da09_o.jpg",
        "https://c1.staticflickr.com/1/1/1053742_34489f1009_o.jpg",
        "https://c1.staticflickr.com/1/1/1053838_46aa549510_o.jpg",
    ]
    read_uris = get_all_image_uris_ordered("tests/test_data/train")
    assert uris == list(read_uris)
