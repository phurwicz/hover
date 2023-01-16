"""
???+ note "Neural network components."

    `torch`-based template classes for implementing neural nets that work the most smoothly with `hover`.
"""
import os
import hover
import numpy as np
import torch
import torch.nn.functional as F
from abc import abstractmethod
from bokeh.models import Slider, CustomJSTickFormatter
from sklearn.metrics import confusion_matrix
from shutil import copyfile
from hover.core import Loggable
from hover.utils.metrics import classification_accuracy
from hover.utils.misc import current_time


class BaseVectorNet(Loggable):

    """
    ???+ note "Abstract transfer learning model defining common signatures."

        Intended to define crucial interactions with built-in recipes like `hover.recipes.active_learning()`.
    """

    @abstractmethod
    def predict_proba(self, inps):
        pass

    @abstractmethod
    def manifold_trajectory(
        self, inps, method=None, reducer_kwargs=None, spline_kwargs=None
    ):
        pass

    @abstractmethod
    def prepare_loader(self, dataset, key, **kwargs):
        pass

    @abstractmethod
    def train(self, train_loader, dev_loader=None, epochs=None, **kwargs):
        pass


class VectorNet(BaseVectorNet):

    """
    ???+ note "Simple transfer learning model: a user-supplied vectorizer followed by a neural net."
        This is a parent class whose children may use different training schemes.

        Coupled with:

        -   `hover.utils.torch_helper.VectorDataset`
    """

    DEFAULT_OPTIM_CLS = torch.optim.Adam
    DEFAULT_OPTIM_LOGLR = 2.0
    DEFAULT_OPTIM_KWARGS = {"lr": 0.1**DEFAULT_OPTIM_LOGLR, "betas": (0.9, 0.999)}

    def __init__(
        self,
        vectorizer,
        architecture,
        state_dict_path,
        labels,
        backup_state_dict=True,
        optimizer_cls=None,
        optimizer_kwargs=None,
        verbose=0,
        example_input="",
    ):
        """
        ???+ note "Create the `VectorNet`, loading parameters if available."

            | Param             | Type       | Description                          |
            | :---------------- | :--------- | :----------------------------------- |
            | `vectorizer`      | `callable` | the feature -> vector function       |
            | `architecture`    | `class`    | a `torch.nn.Module` child class      |
            | `state_dict_path` | `str`      | path to a (could-be-empty) `torch` state dict |
            | `labels`          | `list`     | list of `str` classification labels  |
            | `backup_state_dict` | `bool`   | whether to backup the loaded state dict |
            | `optimizer_cls`   | `subclass of torch.optim.Optimizer` | pytorch optimizer class |
            | `optimizer_kwargs`  | `dict`   | pytorch optimizer kwargs             |
            | `verbose`         | `int`      | logging verbosity level              |
            | `example_input`   | any        | example input to the vectorizer      |
        """

        assert isinstance(
            verbose, int
        ), f"Expected verbose as int, got {type(verbose)} {verbose}"
        self.verbose = verbose
        self.vectorizer = vectorizer
        self.example_input = example_input
        self.architecture = architecture
        self.setup_label_conversion(labels)
        self._dynamic_params = {}

        # set a path to store updated parameters
        self.nn_update_path = state_dict_path

        if backup_state_dict and os.path.isfile(state_dict_path):
            state_dict_backup_path = f"{state_dict_path}.{current_time('%Y%m%d%H%M%S')}"
            copyfile(state_dict_path, state_dict_backup_path)

        # initialize an optimizer object and a dict to hold dynamic parameters
        optimizer_cls = optimizer_cls or self.__class__.DEFAULT_OPTIM_CLS
        optimizer_kwargs = (
            optimizer_kwargs or self.__class__.DEFAULT_OPTIM_KWARGS.copy()
        )

        def callback_reset_nn_optimizer():
            """
            Callback function which has access to optimizer init settings.
            """
            self.nn_optimizer = optimizer_cls(self.nn.parameters())
            assert isinstance(
                self.nn_optimizer, torch.optim.Optimizer
            ), f"Expected an optimizer, got {type(self.nn_optimizer)}"
            self._dynamic_params["optimizer"] = optimizer_kwargs

        self._callback_reset_nn_optimizer = callback_reset_nn_optimizer
        self.setup_nn(use_existing_state_dict=True)
        self._setup_widgets()

    def auto_adjust_setup(self, labels, auto_skip=True):
        """
        ???+ note "Auto-(re)create label encoder/decoder and neural net."

            Intended to be called in and out of the constructor.

            | Param             | Type       | Description                          |
            | :---------------- | :--------- | :----------------------------------- |
            | `labels`          | `list`     | list of `str` classification labels  |
            | `auto_skip`       | `bool`     | skip when labels did not change      |
        """
        # sanity check and skip
        assert isinstance(labels, list), f"Expected a list of labels, got {labels}"
        # if the sequence of labels matches label encoder exactly, skip
        label_match_flag = labels == sorted(
            self.label_encoder.keys(), key=lambda k: self.label_encoder[k]
        )
        if auto_skip and label_match_flag:
            return

        self.setup_label_conversion(labels)
        self.setup_nn(use_existing_state_dict=False)

        self._good(f"adjusted to new list of labels: {labels}")

    def setup_label_conversion(self, labels):
        """
        ???+ note "Set up label encoder/decoder and number of classes."

            | Param             | Type       | Description                          |
            | :---------------- | :--------- | :----------------------------------- |
            | `labels`          | `list`     | list of `str` classification labels  |
        """
        self.label_encoder = {_label: i for i, _label in enumerate(labels)}
        self.label_decoder = {i: _label for i, _label in enumerate(labels)}
        self.num_classes = len(self.label_encoder)

    def setup_nn(self, use_existing_state_dict=True):
        """
        ???+ note "Set up neural network and optimizers."

            Intended to be called in and out of the constructor.

            -   will try to load parameters from state dict by default
            -   option to override and discard previous state dict
                -   often used when the classification targets have changed

            | Param                     | Type       | Description                          |
            | :------------------------ | :--------- | :----------------------------------- |
            | `labels`                  | `list`     | list of `str` classification labels  |
            | `use_existing_state_dict` | `bool`     | whether to use existing state dict   |
        """
        # set up vectorizer and the neural network with appropriate dimensions
        vec_dim = self.vectorizer(self.example_input).shape[0]
        self.nn = self.architecture(vec_dim, self.num_classes)
        self._callback_reset_nn_optimizer()

        state_dict_exists = os.path.isfile(self.nn_update_path)
        # if state dict exists, load it (when consistent) or overwrite
        if state_dict_exists:
            if use_existing_state_dict:
                self.load(self.nn_update_path)
            else:
                self.save(self.nn_update_path)

        self._good(f"reset neural net: in {vec_dim} out {self.num_classes}.")

    def load(self, load_path=None):
        """
        ???+ note "Load neural net parameters if possible."

            Can be directed to a custom state dict.

            | Param       | Type       | Description                  |
            | :---------- | :--------- | :--------------------------- |
            | `load_path` | `str`      | path to a `torch` state dict |
        """
        load_path = load_path or self.nn_update_path
        # if the architecture cannot match the state dict, skip the load and warn
        try:
            self.nn.load_state_dict(torch.load(load_path))
            self._info(f"loaded state dict {load_path}.")
        except Exception as e:
            self._warn(f"load VectorNet state path failed with {type(e)}: {e}")

    @classmethod
    def from_module(cls, model_module, labels, **kwargs):
        """
        ???+ note "Create a VectorNet model from a loadable module."

            | Param          | Type       | Description                          |
            | :------------- | :--------- | :----------------------------------- |
            | `model_module` | `module` or `str` | (path to) a local Python workspace module which contains a get_vectorizer() callable, get_architecture() callable, and a get_state_dict_path() callable |
            | `labels`       | `list`     | list of `str` classification labels  |
            | `**kwargs`     |      | forwarded to `self.__init__()` constructor |
        """
        if isinstance(model_module, str):
            from importlib import import_module

            model_module = import_module(model_module)

        # Load the model by retrieving the inp-to-vec function, architecture, and state dict
        model = cls(
            model_module.get_vectorizer(),
            model_module.get_architecture(),
            model_module.get_state_dict_path(),
            labels,
            **kwargs,
        )

        return model

    def save(self, save_path=None):
        """
        ???+ note "Save the current state dict with authorization to overwrite."
            | Param       | Type  | Description                           |
            | :---------- | :---- | :------------------------------------ |
            | `save_path` | `str` | option alternative path to state dict |
        """
        save_path = save_path or self.nn_update_path
        torch.save(self.nn.state_dict(), save_path)
        verb = "overwrote" if os.path.isfile(save_path) else "saved"
        self._info(f"{verb} state dict {save_path}.")

    def _setup_widgets(self):
        """
        ???+ note "Bokeh widgets for changing hyperparameters through user interaction."
        """
        self.epochs_slider = Slider(start=1, end=50, value=1, step=1, title="# epochs")
        self.loglr_slider = Slider(
            title="learning rate",
            start=1.0,
            end=7.0,
            value=self.__class__.DEFAULT_OPTIM_LOGLR,
            step=0.1,
            format=CustomJSTickFormatter(code="return Math.pow(0.1, tick).toFixed(8)"),
        )

        def update_lr(attr, old, new):
            self._dynamic_params["optimizer"]["lr"] = 0.1**self.loglr_slider.value

        self.loglr_slider.on_change("value", update_lr)

    def _layout_widgets(self):
        """
        ???+ note "Layout of widgets when plotted."
        """
        from bokeh.layouts import row

        return row(self.epochs_slider, self.loglr_slider)

    def view(self):
        """
        ???+ note "Overall layout when plotted."
        """
        return self._layout_widgets()

    def adjust_optimizer_params(self):
        """
        ???+ note "Dynamically change parameters of the neural net optimizer."

            - Intended to be polymorphic in child classes and to be called per epoch.
        """
        for _group in self.nn_optimizer.param_groups:
            _group.update(self._dynamic_params["optimizer"])

    def predict_proba(self, inps):
        """
        ???+ note "End-to-end single/multi-piece prediction from inp to class probabilities."
            | Param  | Type    | Description                          |
            | :----- | :------ | :----------------------------------- |
            | `inps` | dynamic | (a list of) input features to vectorize |
        """
        # if the input is a single piece of inp, cast it to a list
        FLAG_SINGLE = not isinstance(inps, list)
        if FLAG_SINGLE:
            inps = [inps]

        # the actual prediction
        self.nn.eval()
        vectors = torch.Tensor(np.array([self.vectorizer(_inp) for _inp in inps]))
        logits = self.nn(vectors)
        probs = F.softmax(logits, dim=-1).detach().numpy()

        # inverse-cast if applicable
        if FLAG_SINGLE:
            probs = probs[0]

        return probs

    def manifold_trajectory(
        self, inps, method=None, reducer_kwargs=None, spline_kwargs=None
    ):
        """
        ???+ note "Compute a propagation trajectory of the dataset manifold through the neural net."

            1. vectorize inps
            2. forward propagate, keeping intermediates
            3. fit intermediates to N-D manifolds
            4. fit manifolds using Procrustes shape analysis
            5. fit shapes to trajectory splines

            | Param    | Type    | Description                          |
            | :------- | :------ | :----------------------------------- |
            | `inps`   | dynamic | (a list of) input features to vectorize |
            | `method` | `str`   | reduction method: `"umap"` or `"ivis"`  |
            | `reducer_kwargs` | | kwargs to forward to dimensionality reduction |
            | `spline_kwargs` | | kwargs to forward to spline calculation |
        """
        from hover.core.representation.manifold import LayerwiseManifold
        from hover.core.representation.trajectory import manifold_spline

        if method is None:
            method = hover.config["data.embedding"]["default_reduction_method"]

        reducer_kwargs = reducer_kwargs or {}
        spline_kwargs = spline_kwargs or {}

        # step 1 & 2
        vectors = torch.Tensor(np.array([self.vectorizer(_inp) for _inp in inps]))
        self.nn.eval()
        intermediates = self.nn.eval_per_layer(vectors)
        intermediates = [_tensor.detach().numpy() for _tensor in intermediates]

        # step 3 & 4
        LM = LayerwiseManifold(intermediates)
        LM.unfold(method=method, **reducer_kwargs)
        seq_arr, disparities = LM.procrustes()
        seq_arr = np.array(seq_arr)

        # step 5
        traj_arr = manifold_spline(np.array(seq_arr), **spline_kwargs)

        return traj_arr, seq_arr, disparities

    def prepare_loader(self, dataset, key, **kwargs):
        """
        ???+ note "Create dataloader from `SupervisableDataset` with implied vectorizer(s)."

            | Param      | Type  | Description                |
            | :--------- | :---- | :------------------------- |
            | `dataset`  | `hover.core.dataset.SupervisableDataset` | the dataset to load |
            | `key`      | `str` | "train", "dev", or "test"  |
            | `**kwargs` | | forwarded to `dataset.loader()`  |
        """
        return dataset.loader(key, self.vectorizer, **kwargs)

    def train(self, train_loader, dev_loader=None, epochs=None):
        """
        ???+ note "Train the neural network part of the VecNet."

            - intended to be coupled with self.train_batch().

            | Param          | Type         | Description                |
            | :------------- | :----------- | :------------------------- |
            | `train_loader` | `torch.utils.data.DataLoader` | train set |
            | `dev_loader`   | `torch.utils.data.DataLoader` | dev set   |
            | `epochs`       | `int`        | number of epochs to train  |
        """
        epochs = epochs or self.epochs_slider.value

        train_info = []
        for epoch_idx in range(epochs):
            self._dynamic_params["epoch"] = epoch_idx + 1
            self.train_epoch(train_loader)
            if dev_loader is not None:
                dev_loader = train_loader
            acc, conf_mat = self.evaluate(dev_loader)
            train_info.append({"accuracy": acc, "confusion_matrix": conf_mat})
        return train_info

    def train_epoch(self, train_loader, *args, **kwargs):
        """
        ???+ note "Train the neural network for one epoch."

            - Supports flexible args and kwargs for child classes that may implement self.train() and self.train_batch() differently.

            | Param          | Type         | Description                |
            | :------------- | :----------- | :------------------------- |
            | `train_loader` | `torch.utils.data.DataLoader` | train set |
            | `*args`        | | arguments to forward to `train_batch`   |
            | `**kwargs`     | | kwargs to forward to `train_batch`      |
        """
        self.adjust_optimizer_params()
        for batch_idx, (loaded_input, loaded_output, _) in enumerate(train_loader):
            self._dynamic_params["batch"] = batch_idx + 1
            self.train_batch(loaded_input, loaded_output, *args, **kwargs)

    def train_batch(self, loaded_input, loaded_output):
        """
        ???+ note "Train the neural network for one batch."

            | Param           | Type           | Description           |
            | :-------------- | :------------- | :-------------------- |
            | `loaded_input`  | `torch.Tensor` | input tensor          |
            | `loaded_output` | `torch.Tensor` | output tensor         |
        """
        self.nn.train()
        input_tensor = loaded_input.float()
        output_tensor = loaded_output.float()

        # compute logits
        logits = self.nn(input_tensor)
        loss = F.cross_entropy(logits, output_tensor)

        self.nn_optimizer.zero_grad()
        loss.backward()
        self.nn_optimizer.step()

        if self.verbose > 0:
            log_info = dict(self._dynamic_params)
            log_info["performance"] = "Loss {0:.3f}".format(loss)
            self._print(
                "{0: <80}".format(
                    "Train: Epoch {epoch} Batch {batch} {performance}".format(
                        **log_info
                    )
                ),
                end="\r",
            )

    def evaluate(self, dev_loader):
        """
        ???+ note "Evaluate the VecNet against a dev set."

            | Param        | Type         | Description                |
            | :----------- | :----------- | :------------------------- |
            | `dev_loader` | `torch.utils.data.DataLoader` | dev set   |
        """
        self.nn.eval()
        true = []
        pred = []
        for loaded_input, loaded_output, _idx in dev_loader:
            _input_tensor = loaded_input.float()
            _output_tensor = loaded_output.float()

            _logits = self.nn(_input_tensor)
            _true_batch = _output_tensor.argmax(dim=1).detach().numpy()
            _pred_batch = F.softmax(_logits, dim=1).argmax(dim=1).detach().numpy()
            true.append(_true_batch)
            pred.append(_pred_batch)
        true = np.concatenate(true)
        pred = np.concatenate(pred)
        accuracy = classification_accuracy(true, pred)
        conf_mat = confusion_matrix(true, pred)

        if self.verbose >= 0:
            log_info = dict(self._dynamic_params)
            log_info["performance"] = "Acc {0:.3f}".format(accuracy)
            self._info(
                "{0: <80}".format(
                    "Eval: Epoch {epoch} {performance}".format(**log_info)
                )
            )

        return accuracy, conf_mat
