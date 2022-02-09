import pytest
import numpy as np
import uuid
from copy import deepcopy
from hover.core.neural import VectorNet, MultiVectorNet


@pytest.fixture
def example_vecnet_args(mini_supervisable_text_dataset):
    module_name = "fixture_module.vector_net"
    target_labels = mini_supervisable_text_dataset.classes[:]
    return (module_name, target_labels)


@pytest.fixture
def blank_vecnet():
    model = VectorNet.from_module("fixture_module.vector_net", [], verbose=10)
    return model


@pytest.fixture
def example_vecnet(example_vecnet_args):
    model = VectorNet.from_module(*example_vecnet_args, verbose=10)
    return model


def create_new_multivecnet(target_labels):
    module_names = [
        "fixture_module.multi_vector_net.model1",
        "fixture_module.multi_vector_net.model2",
        "fixture_module.multi_vector_net.model3",
        "fixture_module.multi_vector_net.model4",
    ]
    nets = [
        VectorNet.from_module(_m, target_labels, backup_state_dict=False)
        for _m in module_names
    ]
    return MultiVectorNet(nets, primary=0)


@pytest.fixture
def example_multivecnet(mini_supervisable_text_dataset):
    target_labels = mini_supervisable_text_dataset.classes
    return create_new_multivecnet(target_labels)


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
    def test_auto_adjust_setup(blank_vecnet, mini_supervisable_text_dataset):
        targets = mini_supervisable_text_dataset.classes
        old_classes = sorted(
            blank_vecnet.label_encoder.keys(),
            key=lambda k: blank_vecnet.label_encoder[k],
        )
        old_nn = blank_vecnet.nn
        # normal change of classes should create a new NN
        blank_vecnet.auto_adjust_setup(targets)
        first_nn = blank_vecnet.nn
        assert first_nn is not old_nn
        # identical classes should trigger autoskip
        blank_vecnet.auto_adjust_setup(targets)
        second_nn = blank_vecnet.nn
        assert second_nn is first_nn
        # change of class order should create a new NN
        blank_vecnet.auto_adjust_setup(targets[1:] + targets[:1])
        third_nn = blank_vecnet.nn
        assert third_nn is not second_nn
        blank_vecnet.auto_adjust_setup(old_classes)

    @staticmethod
    @pytest.mark.lite
    def test_adjust_optimier_params(example_vecnet):
        example_vecnet.adjust_optimizer_params()

    @staticmethod
    @pytest.mark.lite
    def test_predict_proba(example_vecnet, mini_supervisable_text_dataset):
        subroutine_predict_proba(example_vecnet, mini_supervisable_text_dataset)

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
    @pytest.mark.lite
    def test_basics(example_multivecnet):
        # use unique paths to avoid unwanted interaction with other tests
        for _net in example_multivecnet.vector_nets:
            _uuid = str(uuid.uuid1())
            _nn_path = _net.nn_update_path.replace(".pt", f"-{_uuid}.pt")
            _net.nn_update_path = _nn_path
        example_multivecnet.save()

        # TODO: make tests below more meaningful
        _ = example_multivecnet.view()

    @staticmethod
    @pytest.mark.lite
    def test_predict_proba(example_multivecnet, mini_supervisable_text_dataset):
        subroutine_predict_proba(example_multivecnet, mini_supervisable_text_dataset)

    @staticmethod
    def test_manifold_trajectory(example_multivecnet, mini_supervisable_text_dataset):
        dataset = mini_supervisable_text_dataset.copy()
        inps = dataset.dfs["raw"]["text"].tolist()[:100]
        _ = example_multivecnet.manifold_trajectory(inps)

    @staticmethod
    def test_train_and_evaluate(noisy_supervisable_text_dataset):
        """
        Verify that MultiVectorNet can be used for denoising a noised dataset.
        """
        # create two MultiVectorNets with the same setup
        dataset = noisy_supervisable_text_dataset.copy()
        multi_a = create_new_multivecnet(dataset.classes)
        multi_b = create_new_multivecnet(dataset.classes)

        # prepare multi-vector loaders
        vectorizers = [_net.vectorizer for _net in multi_a.vector_nets]
        train_loader = dataset.loader(
            "train", *vectorizers, smoothing_coeff=0.0, batch_size=64
        )
        dev_loader = dataset.loader("dev", *vectorizers)
        test_loader = dataset.loader("test", *vectorizers)

        # use one MultiVectorNet for denoising treatment and the other for control
        kwargs_a = dict(
            warmup_epochs=40,
            warmup_noise=0.0,
            warmup_lr=0.05,
            warmup_momentum=0.9,
            postwm_epochs=40,
            postwm_noise=0.5,
            postwm_lr=0.01,
            postwm_momentum=0.7,
        )
        kwargs_b = dict(
            warmup_epochs=40,
            warmup_noise=0.0,
            warmup_lr=0.05,
            warmup_momentum=0.9,
            postwm_epochs=40,
            postwm_noise=0.0,
            postwm_lr=0.01,
            postwm_momentum=0.7,
        )

        # train both MultiVectorNets
        train_info_a = multi_a.train(train_loader, dev_loader=dev_loader, **kwargs_a)
        train_info_b = multi_b.train(train_loader, dev_loader=dev_loader, **kwargs_b)
        for _train_info in [train_info_a, train_info_b]:
            assert isinstance(_train_info, list)
            assert isinstance(_train_info[0], dict)

        # evaluate both MultiVectorNets
        accuracy_a, conf_mat_a = multi_a.evaluate_ensemble(test_loader)
        accuracy_b, conf_mat_b = multi_b.evaluate_ensemble(test_loader)
        assert isinstance(accuracy_a, float) and isinstance(accuracy_b, float)
        assert isinstance(conf_mat_a, np.ndarray) and isinstance(conf_mat_b, np.ndarray)
        assert (
            accuracy_a > accuracy_b + 1e-3
        ), f"Expected denoising to achieve better ensemble accuracy (> 0.001 margin) on a noised dataset, got {accuracy_a} (treatment) vs. {accuracy_b} (control)"

        # evaluate individual VectorNets
        acc_list_a, _, _ = multi_a.evaluate_individual(test_loader)
        acc_list_b, _, _ = multi_b.evaluate_individual(test_loader)
        mean_acc_a, mean_acc_b = np.mean(acc_list_a), np.mean(acc_list_b)
        assert (
            mean_acc_a > mean_acc_b + 5e-3
        ), f"Expected denoising to achieve better mean individual accuracy (> 0.005 margin) on a noised dataset, got {mean_acc_a} (treatment) vs. {mean_acc_b} (control)"
