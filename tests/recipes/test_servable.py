import time
import pytest
from hover.recipes.stable import simple_annotator, linked_annotator
from hover.recipes.experimental import active_learning, snorkel_crosscheck
from bokeh.server.server import Server


@pytest.mark.lite
def test_builtin_servable_recipes(
    mini_supervisable_text_dataset_embedded,
    dummy_vecnet_callback,
    dummy_labeling_function_list,
):
    dataset = mini_supervisable_text_dataset_embedded.copy()
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
    server = Server(app_dict)
    server.start()
    time.sleep(20)
    server.stop()
