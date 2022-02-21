from hover.utils.datasets import newsgroups_dictl, newsgroups_reduced_dictl
import pytest


@pytest.mark.lite
def test_20_newsgroups():
    for dictl_method, num_classes in [
        (newsgroups_dictl, 20),
        (newsgroups_reduced_dictl, 7),
    ]:
        my_20ng, label_encoder, label_decoder = dictl_method()

        assert isinstance(my_20ng, dict)
        for _key in ["train", "test"]:
            assert isinstance(my_20ng["train"], list)
            assert isinstance(my_20ng["train"][0], dict)
            assert isinstance(my_20ng["train"][0]["label"], str)
            assert isinstance(my_20ng["train"][0]["text"], str)

        assert isinstance(label_encoder, dict)
        assert isinstance(label_decoder, dict)
        assert len(label_encoder) == num_classes + 1
        assert len(label_decoder) == num_classes + 1
        assert set(label_encoder.keys()) == set(label_decoder.values())
        assert set(label_decoder.keys()) == set(label_encoder.values())
