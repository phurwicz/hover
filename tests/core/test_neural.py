import pytest
import numpy as np
from copy import deepcopy
from hover.core.neural import VectorNet, MultiVectorNet
from hover.utils.denoising import identity_adjacency, cyclic_except_last


@pytest.fixture
def example_vecnet_args(mini_supervisable_text_dataset):
    module_name = "fixture_module.vector_net"
    target_labels = mini_supervisable_text_dataset.classes[:]
    return (module_name, target_labels)


@pytest.fixture
def example_vecnet(example_vecnet_args):
    model = VectorNet.from_module(*example_vecnet_args)
    return model


@pytest.fixture
def example_new_multivecnet(noisy_supervisable_text_dataset):
    target_labels = noisy_supervisable_text_dataset.classes[:]
    module_names = [
        "fixture_module.multi_vector_net.model1",
        "fixture_module.multi_vector_net.model2",
        "fixture_module.multi_vector_net.model3",
        "fixture_module.multi_vector_net.model4",
    ]

    def callback():
        return MultiVectorNet(
            [VectorNet.from_module(_m, target_labels) for _m in module_names]
        )

    return callback


@pytest.fixture
def example_multivecnet(example_new_multivecnet):
    return example_new_multivecnet()


@pytest.mark.core
class TestVectorNet(object):
    """
    For the VectorNet base class.
    """

    @staticmethod
    def test_save_and_load(example_vecnet, example_vecnet_args):
        default_path = example_vecnet.nn_update_path
        example_vecnet.save(f"{default_path}.test")
        loaded_vecnet = VectorNet.from_module(*example_vecnet_args)
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
        dev_loader = dataset.loader("dev", example_vecnet.vectorizer)
        test_loader = dataset.loader("test", example_vecnet.vectorizer)

        train_info = vecnet.train(dev_loader, dev_loader, epochs=5)
        accuracy, conf_mat = vecnet.evaluate(test_loader)

        assert isinstance(train_info, list)
        assert isinstance(train_info[0], dict)
        assert isinstance(accuracy, float)
        assert isinstance(conf_mat, np.ndarray)


@pytest.mark.core
class TestMultiVectorNet(object):
    """
    For the MultiVectorNet class.
    """

    @staticmethod
    def test_adjust_optimier_params(example_multivecnet):
        example_vecnet.adjust_optimizer_params()

    @staticmethod
    def test_train_and_evaluate(example_multivecnet, noisy_supervisable_text_dataset):
        """
        Verify that MultiVectorNet can be used for denoising a noised dataset.
        """
        # create two MultiVectorNets with the same setup
        multi_a = example_new_multivecnet()
        multi_b = example_new_multivecnet()
        dataset = noisy_supervisable_text_dataset

        # prepare multi-vector loaders
        vectorizers = [_net.vectorizer for _net in multi_a.vector_nets]
        train_loader = dataset.loader(
            "train", *vectorizers, smoothing_coeff=0.1, batch_size=256
        )
        dev_loader = dataset.loader("dev", *vectorizers)
        test_loader = dataset.loader("test", *vectorizers)

        # use one MultiVectorNet for denoising treatment and the other for control
        def get_params(warmup_epochs=5, coteach_epochs=10, forget_rate=0.3):
            for i in range(warmup_epochs):
                yield {
                    "forget_rate": 0.0,
                    "optimizer": [{"lr": 0.1, "momentum": 0.9}] * 4,
                    "adjacency_function": identity_adjacency,
                }
            for i in range(coteach_epochs):
                yield {
                    "forget_rate": forget_rate,
                    "optimizer": [{"lr": 0.01, "momentum": 0.9}] * 4,
                    "adjacency_function": cyclic_except_last,
                }

        param_a = get_params(warmup_epochs=5, coteach_epochs=10, forget_rate=0.5)
        param_b = get_params(warmup_epochs=15, coteach_epochs=0, forget_rate=0.0)

        # train both MultiVectorNets
        train_info_a = multi_a.train(train_loader, param_a, dev_loader=dev_loader)
        train_info_b = multi_b.train(train_loader, param_b, dev_loader=dev_loader)
        for _train_info in [train_info_a, train_info_b]:
            assert isinstance(_train_info, list)
            assert isinstance(_train_info[0], dict)

        # evaluate both MultiVectorNets
        accuracy_a, conf_mat_a = multi_a.evaluate_ensemble(test_loader)
        accuracy_b, conf_mat_b = multi_b.evaluate_ensemble(test_loader)
        assert isinstance(accuracy_a, float) and isinstance(accuracy_b, float)
        assert isinstance(conf_mat_a, np.ndarray) and isinstance(conf_mat_b, np.ndarray)
        assert (
            accuracy_a > accuracy_b
        ), f"Expected denoising to achieve better accuracy on a noised dataset, got {accuracy_a} (treatment) vs. {accuracy_b} (control)"
