import pytest
import numpy as np
from copy import deepcopy
from hover.core.neural import VectorNet


@pytest.fixture
def example_vecnet():
    model = VectorNet.from_module("model_template", ["positive", "negative"])
    return model


@pytest.mark.core
class TestVectorNet(object):
    """
    For the VectorNet base class.
    """

    @staticmethod
    def test_save_and_load(example_vecnet):
        default_path = example_vecnet.nn_update_path
        example_vecnet.save(f"{default_path}.test")
        loaded_vecnet = VectorNet.from_module(
            "model_template", ["positive", "negative"]
        )
        loaded_vecnet.save()

    @staticmethod
    def test_adjust_optimier_params(example_vecnet):
        example_vecnet.adjust_optimizer_params()

    @staticmethod
    def test_predict_proba(example_vecnet):
        proba_single = example_vecnet.predict_proba("hello")
        assert proba_single.shape[0] == 2
        proba_multi = example_vecnet.predict_proba(["hello", "bye", "ciao"])
        assert proba_multi.shape[0] == 3
        assert proba_multi.shape[1] == 2

    @staticmethod
    def test_manifold_trajectory(example_vecnet, mini_df_text):
        for _method in ["umap", "ivis"]:
            traj_arr, seq_arr, disparities = example_vecnet.manifold_trajectory(
                mini_df_text["text"].tolist()
            )
            assert isinstance(traj_arr, np.ndarray)
            assert isinstance(seq_arr, np.ndarray)
            assert isinstance(disparities, list)
            assert isinstance(disparities[0], float)

    @staticmethod
    def test_train_and_evaluate(example_vecnet, mini_supervisable_text_dataset):
        vecnet = deepcopy(example_vecnet)
        dataset = mini_supervisable_text_dataset
        dev_loader = dataset.loader("dev", vectorizer=example_vecnet.vectorizer)
        test_loader = dataset.loader("test", vectorizer=example_vecnet.vectorizer)

        train_info = vecnet.train(dev_loader, dev_loader, epochs=5)
        accuracy, conf_mat = vecnet.evaluate(test_loader)

        assert isinstance(train_info, list)
        assert isinstance(train_info[0], dict)
        assert isinstance(accuracy, float)
        assert isinstance(conf_mat, np.ndarray)
