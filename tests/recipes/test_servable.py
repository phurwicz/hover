import pytest
import requests
from hover.recipes.stable import simple_annotator, linked_annotator
from hover.recipes.experimental import active_learning, snorkel_crosscheck
from bokeh.server.server import Server


@pytest.mark.lite
def test_builtin_servable_recipes(
    example_text_dataset,
    dummy_vecnet_callback,
    dummy_labeling_function_list,
):
    dataset = example_text_dataset.copy()
    vecnet = dummy_vecnet_callback(dataset)
    simple = simple_annotator(dataset)
    linked = linked_annotator(dataset)
    active = active_learning(dataset, vecnet)
    snorkel = snorkel_crosscheck(dataset, dummy_labeling_function_list)
    app_dict = {
        "simple": simple,
        "linked": linked,
        "active": active,
        "snorkel": snorkel,
    }
    server = Server(app_dict, port=5007)
    server.start()
    for _app in app_dict.keys():
        _response = requests.get(f"http://localhost:5007/{_app}")
        assert _response.status_code == 200
    server.stop()
