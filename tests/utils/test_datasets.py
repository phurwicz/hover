from hover.utils.datasets import newsgroups_dictl


def test_20_newsgroups():
    my_20ng, label_encoder, label_decoder = newsgroups_dictl()

    assert isinstance(my_20ng, dict)
    for _key in ["train", "test"]:
        assert isinstance(my_20ng["train"], list)
        assert isinstance(my_20ng["train"][0], dict)
        assert isinstance(my_20ng["train"][0]["label"], str)
        assert isinstance(my_20ng["train"][0]["text"], str)

    assert isinstance(label_encoder, dict)
    assert isinstance(label_decoder, dict)
    assert len(label_encoder) == 21
    assert len(label_decoder) == 21
    assert set(label_encoder.keys()) == set(label_decoder.values())
    assert set(label_decoder.keys()) == set(label_encoder.values())
