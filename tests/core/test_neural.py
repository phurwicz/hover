import pytest
from hover.core.neural import create_vector_net_from_module as create_tvnet, VectorNet


@pytest.fixture
def example_tvnet():
    model = create_tvnet(VectorNet, "model_template", ["positive", "negative"])
    return model


@pytest.mark.core
class TestVectorNet(object):
    """
    For the VectorNet base class.
    """

    @staticmethod
    def test_save_and_load(example_tvnet):
        default_path = example_tvnet.nn_update_path
        example_tvnet.save(f"{default_path}.test")
        loaded_tvnet = create_tvnet(
            VectorNet, "model_template", ["positive", "negative"]
        )
        loaded_tvnet.save()

    @staticmethod
    def test_adjust_optimier_params(example_tvnet):
        example_tvnet.adjust_optimizer_params()

    @staticmethod
    def test_predict_proba(example_tvnet):
        proba_single = example_tvnet.predict_proba("hello")
        assert proba_single.shape[0] == 2
        proba_multi = example_tvnet.predict_proba(["hello", "bye", "ciao"])
        assert proba_multi.shape[0] == 3
        assert proba_multi.shape[1] == 2

    @staticmethod
    def test_manifold_trajectory(example_tvnet, mini_df_text):
        traj_arr, seq_arr, disparities = example_tvnet.manifold_trajectory(
            mini_df_text["text"].tolist()
        )
