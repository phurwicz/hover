import pytest
import numpy as np
from copy import deepcopy
from hover.core.neural import VectorNet
from hover.module_config import DataFrame


@pytest.fixture
def example_vecnet_args(example_text_dataset):
    module_name = "fixture_module.text_vector_net"
    target_labels = example_text_dataset.classes[:]
    return (module_name, target_labels)


@pytest.fixture
def blank_vecnet():
    model = VectorNet.from_module("fixture_module.text_vector_net", [], verbose=10)
    return model


@pytest.fixture
def example_vecnet(example_vecnet_args):
    model = VectorNet.from_module(*example_vecnet_args, verbose=10)
    return model


def subroutine_predict_proba(net, dataset):
    num_classes = len(dataset.classes)
    proba_single = net.predict_proba("hello")
    assert proba_single.shape[0] == num_classes
    proba_multi = net.predict_proba(["hello", "bye", "ciao"])
    assert proba_multi.shape[0] == 3
    assert proba_multi.shape[1] == num_classes


@pytest.mark.core
class TestVectorNet(object):
    """
    For the VectorNet base class.
    """

    @staticmethod
    @pytest.mark.lite
    def test_save_and_load(example_vecnet, example_vecnet_args):
        default_path = example_vecnet.nn_update_path
        example_vecnet.save(f"{default_path}.test")
        loaded_vecnet = VectorNet.from_module(*example_vecnet_args)
        loaded_vecnet.save()

    @staticmethod
    @pytest.mark.lite
    def test_auto_adjust_setup(blank_vecnet, example_text_dataset):
        vecnet = deepcopy(blank_vecnet)
        targets = example_text_dataset.classes
        old_classes = sorted(
            vecnet.label_encoder.keys(),
            key=lambda k: vecnet.label_encoder[k],
        )
        old_nn = vecnet.nn
        # normal change of classes should create a new NN
        vecnet.auto_adjust_setup(targets)
        first_nn = vecnet.nn
        assert first_nn is not old_nn
        # identical classes should trigger autoskip
        vecnet.auto_adjust_setup(targets)
        second_nn = vecnet.nn
        assert second_nn is first_nn
        # change of class order should create a new NN
        vecnet.auto_adjust_setup(targets[1:] + targets[:1])
        third_nn = vecnet.nn
        assert third_nn is not second_nn
        vecnet.auto_adjust_setup(old_classes)

    @staticmethod
    @pytest.mark.lite
    def test_adjust_optimier_params(example_vecnet):
        example_vecnet.adjust_optimizer_params()

    @staticmethod
    @pytest.mark.lite
    def test_predict_proba(example_vecnet, example_text_dataset):
        subroutine_predict_proba(example_vecnet, example_text_dataset)

    @staticmethod
    def test_manifold_trajectory(example_vecnet, example_raw_df):
        for _method in ["umap", "ivis"]:
            traj_arr, seq_arr, disparities = example_vecnet.manifold_trajectory(
                DataFrame.series_tolist(example_raw_df["text"])
            )
            assert isinstance(traj_arr, np.ndarray)
            assert isinstance(seq_arr, np.ndarray)
            assert isinstance(disparities, list)
            assert isinstance(disparities[0], float)

    @staticmethod
    def test_train_and_evaluate(example_vecnet, example_text_dataset):
        vecnet = deepcopy(example_vecnet)
        dataset = example_text_dataset
        dev_loader = dataset.loader("dev", example_vecnet.vectorizer)
        test_loader = dataset.loader("test", example_vecnet.vectorizer)

        train_info = vecnet.train(dev_loader, dev_loader, epochs=5)
        accuracy, conf_mat = vecnet.evaluate(test_loader)

        assert isinstance(train_info, list)
        assert isinstance(train_info[0], dict)
        assert isinstance(accuracy, float)
        assert isinstance(conf_mat, np.ndarray)
