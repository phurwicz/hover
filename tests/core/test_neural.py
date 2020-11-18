import pytest
from hover.core.neural import create_vector_net_from_module as create_tvnet, VectorNet


@pytest.fixture
def example_tvnet():
    model = create_tvnet(VectorNet, "model_template", ["positive", "negative"])
    return model


class TestVectorNet(object):
    """
    For the VectorNet base class.
    """

    def test_save(self, example_tvnet):
        default_path = example_tvnet.nn_update_path
        example_tvnet.save(f"{default_path}.test")

    def test_adjust_optimier_params(self, example_tvnet):
        example_tvnet.adjust_optimizer_params()

    def test_predict_proba(self, example_tvnet):
        proba_single = example_tvnet.predict_proba("hello")
        assert proba_single.shape[0] == 2
        proba_multi = example_tvnet.predict_proba(["hello", "bye", "ciao"])
        assert proba_multi.shape[0] == 3
        assert proba_multi.shape[1] == 2
