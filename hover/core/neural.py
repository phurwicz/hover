"""
???+ note "Neural network components."

    `torch`-based template classes for implementing neural nets that work the most smoothly with `hover`
"""
import torch
import torch.nn.functional as F
from hover.core import Loggable
from hover.utils.copied.snorkel import cross_entropy_with_probs
from hover.utils.metrics import classification_accuracy
from hover.utils.misc import current_time
from hover.utils.denoising import (
    loss_coteaching_graph,
    prediction_disagreement,
    accuracy_priority,
    identity_adjacency,
)
from abc import abstractmethod
from bokeh.models import Slider, FuncTickFormatter
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from shutil import copyfile


class BaseVectorNet(Loggable):

    """
    ???+ note "Abstract transfer learning model defining common signatures."

        Intended to define crucial interactions with built-in recipes like `hover.recipes.active_learning()`.
    """

    @abstractmethod
    def predict_proba(self, inps):
        pass

    @abstractmethod
    def manifold_trajectory(self, inps, method="umap", **kwargs):
        pass

    @abstractmethod
    def prepare_loader(self, dataset, key, **kwargs):
        pass

    @abstractmethod
    def train(self, train_loader, dev_loader=None, **kwargs):
        pass


class VectorNet(BaseVectorNet):

    """
    ???+ note "Simple transfer learning model: a user-supplied vectorizer followed by a neural net."
        This is a parent class whose children may use different training schemes.

        Coupled with:

        -   `hover.utils.torch_helper.VectorDataset`
    """

    DEFAULT_OPTIM_CLS = torch.optim.Adam
    DEFAULT_OPTIM_KWARGS = {"lr": 0.01, "betas": (0.9, 0.999)}

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

    def _layout_widgets(self):
        """
        ???+ note "Layout of widgets when plotted."
        """
        from bokeh.layouts import column

        return column(self.epochs_slider)

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
        vectors = torch.Tensor([self.vectorizer(_inp) for _inp in inps])
        logits = self.nn(vectors)
        probs = F.softmax(logits, dim=-1).detach().numpy()

        # inverse-cast if applicable
        if FLAG_SINGLE:
            probs = probs[0]

        return probs

    def manifold_trajectory(self, inps, method="umap", **kwargs):
        """
        ???+ note "Compute a propagation trajectory of the dataset manifold through the neural net."

            1. vectorize inps
            2. forward propagate, keeping intermediates
            3. fit intermediates to 2D manifolds
            4. fit manifolds using Procrustes shape analysis
            5. fit shapes to trajectory splines

            | Param    | Type    | Description                          |
            | :------- | :------ | :----------------------------------- |
            | `inps`   | dynamic | (a list of) input features to vectorize |
            | `method` | `str`   | reduction method: `"umap"` or `"ivis"`  |
            | `**kwargs` | | kwargs to forward to dimensionality reduction |
        """
        from hover.core.representation.manifold import LayerwiseManifold
        from hover.core.representation.trajectory import manifold_spline

        # step 1 & 2
        vectors = torch.Tensor([self.vectorizer(_inp) for _inp in inps])
        self.nn.eval()
        intermediates = self.nn.eval_per_layer(vectors)
        intermediates = [_tensor.detach().numpy() for _tensor in intermediates]

        # step 3 & 4
        LM = LayerwiseManifold(intermediates)
        LM.unfold(method=method)
        seq_arr, disparities = LM.procrustes()
        seq_arr = np.array(seq_arr)

        # step 5
        traj_arr = manifold_spline(np.array(seq_arr), **kwargs)

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

            - This method is a vanilla template and is intended to be overridden in child classes.
            - Also intended to be coupled with self.train_batch().

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
        loss = cross_entropy_with_probs(logits, output_tensor)

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


class MultiVectorNet(BaseVectorNet):

    """
    ???+ note "Ensemble transfer learning model: multiple jointly-trained VectorNet's."

        One of the `VectorNet`s is treated as the "primary". Functionalities that only
        work on single `VectorNet`s will point to the primary instead.

        Note that the `VectorNet`s can have different vectorizers.
        Consequently, when training the nets, they expect multiple vectors per input.

        Coupled with:

        -   `hover.utils.torch_helper.MultiVectorDataset`
    """

    DEFAULT_ADJACENCY_FUNC = accuracy_priority

    def __init__(self, vector_nets, primary=0, verbose=0):
        """
        ???+ note "Create the `VectorNet`, loading parameters if available."
            | Param           | Type   | Description                  |
            | :-------------- | :----- | :--------------------------- |
            | `vector_nets`   | `list` | list of VectorNet instances  |
            | `primary`       | `int`  | index of the primary VectorNet |
            | `verbose`       | `int`  | logging verbosity level      |
        """
        assert isinstance(
            primary, int
        ), f"Expected primary VectorNet index as int, got {type(primary)} {primary}"
        assert primary < len(
            vector_nets
        ), f"Primary VectorNet index out of range ({primary} in a list of {len(vector_nets)})"
        self.vector_nets = vector_nets
        self.primary = primary
        self._dynamic_params = dict()
        assert isinstance(
            verbose, int
        ), f"Expected verbose as int, got {type(verbose)} {verbose}"
        self.verbose = verbose
        self._setup_widgets()
        self._warn(
            "this class is in preview and is not sufficiently tested. Use with caution."
        )

    def auto_adjust_setup(self, labels):
        """
        ???+ note "Auto-(re)create label encoder/decoder and neural net."
            | Param             | Type       | Description                          |
            | :---------------- | :--------- | :----------------------------------- |
            | `labels`          | `list`     | list of `str` classification labels  |
        """
        for _net in self.vector_nets:
            _net.auto_adjust_setup(labels)

    def save(self, save_path=None):
        """
        ???+ note "Save the current state dict with authorization to overwrite."
            | Param       | Type  | Description                           |
            | :---------- | :---- | :------------------------------------ |
            | `save_path` | `str` | option alternative path to state dict |
        """
        for _net in self.vector_nets:
            if save_path is not None:
                self._warn(
                    "save_path is ignored. Please specify it on a single VectorNet."
                )
            torch.save(_net.nn.state_dict(), _net.nn_update_path)

    def _setup_widgets(self):
        """
        ???+ note "Bokeh widgets for changing hyperparameters through user interaction."
        """
        epochs_shared_kwargs = dict(start=1, end=50, value=1, step=1)
        self.warmup_epochs_slider = Slider(
            title="# warmup epochs", **epochs_shared_kwargs
        )
        self.postwm_epochs_slider = Slider(
            title="# post-warmup epochs", **epochs_shared_kwargs
        )

        noise_shared_kwargs = dict(start=0.0, end=0.5, step=0.01)
        self.warmup_noise_slider = Slider(
            value=0.0, title="# warmup denoise rate", **noise_shared_kwargs
        )
        self.postwm_noise_slider = Slider(
            value=0.1, title="# post-warmup denoise rate", **noise_shared_kwargs
        )

        lr_shared_kwargs = dict(
            start=0.0,
            end=7.0,
            value=1.0,
            step=0.1,
            format=FuncTickFormatter(code="return Math.pow(0.1, tick).toFixed(8)"),
        )
        self.warmup_loglr_slider = Slider(
            title="# warmup learning rate", **lr_shared_kwargs
        )
        self.postwm_loglr_slider = Slider(
            title="# post-warmup learning rate", **lr_shared_kwargs
        )

        momentum_shared_kwargs = dict(start=0.0, end=1.0, step=0.01)
        self.warmup_momentum_slider = Slider(
            value=0.9, title="# warmup momentum", **momentum_shared_kwargs
        )
        self.postwm_momentum_slider = Slider(
            value=0.7, title="# post-warmup momentum", **momentum_shared_kwargs
        )

    def _layout_widgets(self):
        """
        ???+ note "Layout of widgets when plotted."
        """
        from bokeh.layouts import row, column

        layout = row(
            column(
                self.warmup_epochs_slider,
                self.warmup_noise_slider,
                self.warmup_loglr_slider,
                self.warmup_momentum_slider,
            ),
            column(
                self.postwm_epochs_slider,
                self.postwm_noise_slider,
                self.postwm_loglr_slider,
                self.postwm_momentum_slider,
            ),
        )
        return layout

    def view(self):
        """
        ???+ note "Overall layout when plotted."
        """
        return self._layout_widgets()

    def predict_proba(self, inps):
        """
        ???+ note "End-to-end single/multi-piece prediction from inp to class probabilities."

            Currently only uses the primary VectorNet.
            *Dev note*: consider combining logits.

            | Param  | Type    | Description                          |
            | :----- | :------ | :----------------------------------- |
            | `inps` | dynamic | (a list of) input features to vectorize |
        """
        vecnet = self.vector_nets[self.primary]
        return vecnet.predict_proba(inps)

    def manifold_trajectory(self, inps, method="umap", **kwargs):
        """
        ???+ note "Compute a propagation trajectory of the dataset manifold through the neural net."

            Currently only uses the primary VectorNet.
            *Dev note*: consider padding and concatenation to align intermediate layers.

            | Param    | Type    | Description                          |
            | :------- | :------ | :----------------------------------- |
            | `inps`   | dynamic | (a list of) input features to vectorize |
            | `method` | `str`   | reduction method: `"umap"` or `"ivis"`  |
            | `**kwargs` | | kwargs to forward to dimensionality reduction |
        """
        vecnet = self.vector_nets[self.primary]
        return vecnet.manifold_trajectory(inps, method=method, **kwargs)

    def prepare_loader(self, dataset, key, **kwargs):
        """
        ???+ note "Create dataloader from `SupervisableDataset` with implied vectorizer(s)."

            | Param      | Type  | Description                |
            | :--------- | :---- | :------------------------- |
            | `dataset`  | `hover.core.dataset.SupervisableDataset` | the dataset to load |
            | `key`      | `str` | "train", "dev", or "test"  |
            | `**kwargs` | | forwarded to `dataset.loader()`  |
        """
        vectorizers = [_net.vectorizer for _net in self.vector_nets]
        return dataset.loader(key, *vectorizers, **kwargs)

    def hyperparam_config_from_widgets(self):
        """
        ???+ note "Get training hyperparameter configuration from widgets."
        """
        config = {
            "warmup_epochs": self.warmup_epochs_slider.value,
            "warmup_noise": self.warmup_noise_slider.value,
            "warmup_lr": 0.1**self.warmup_loglr_slider.value,
            "warmup_momentum": self.warmup_momentum_slider.value,
            "postwm_epochs": self.postwm_epochs_slider.value,
            "postwm_noise": self.postwm_noise_slider.value,
            "postwm_lr": 0.1**self.postwm_loglr_slider.value,
            "postwm_momentum": self.postwm_momentum_slider.value,
        }
        return config

    def hyperparam_per_epoch(self, **kwargs):
        """
        ???+ note "Produce dynamic hyperparameters from widget and overrides."

            | Param      | Type  | Description                |
            | :--------- | :---- | :------------------------- |
            | `**kwargs` | | forwarded to `dataset.loader()`  |
        """
        config = self.hyperparam_config_from_widgets()
        config.update(kwargs)

        for _ in range(config["warmup_epochs"]):
            yield {
                "denoise_rate": config["warmup_noise"],
                "optimizer": [
                    {
                        "lr": config["warmup_lr"],
                        "momentum": config["warmup_momentum"],
                    }
                ],
            }
        for _ in range(config["postwm_epochs"]):
            yield {
                "denoise_rate": config["postwm_noise"],
                "optimizer": [
                    {
                        "lr": config["postwm_lr"],
                        "momentum": config["postwm_momentum"],
                    }
                ],
            }

    def train(
        self,
        train_loader,
        dev_loader=None,
        **kwargs,
    ):
        """
        ???+ note "Train multiple VectorNet's jointly."

            | Param          | Type         | Description                |
            | :------------- | :----------- | :------------------------- |
            | `train_loader` | `torch.utils.data.DataLoader` | train set |
            | `dev_loader`   | `torch.utils.data.DataLoader` | dev set   |
            | `**kwargs`     | | forwarded to `self.hyperparam_per_epoch()`  |
            ```
        """
        params_per_epoch = self.hyperparam_per_epoch(**kwargs)
        train_info = []
        prev_best_acc = 0.0
        for epoch_idx, param_dict in enumerate(params_per_epoch):
            self._dynamic_params["epoch"] = epoch_idx + 1
            self._dynamic_params.update(param_dict)
            self.train_epoch(train_loader)
            if dev_loader is None:
                dev_loader = train_loader
            acc_list, conf_list, disagree_rate = self.evaluate_individual(dev_loader)

            adj_func = self._dynamic_params.get(
                "adjacency_function",
                self.__class__.DEFAULT_ADJACENCY_FUNC,
            )

            # keep training information and re-pick model teachers
            info_dict = {
                "accuracy": acc_list,
                "confusion_matrix": conf_list,
                "disagreement_rate": disagree_rate,
            }
            train_info.append(info_dict)

            # prepare training detail for the next epoch
            epoch_best_acc = np.max(acc_list)
            # epoch_best_idx = np.argmax(acc_list)
            coteaching_flag = self._dynamic_params["denoise_rate"] > 0.0
            if coteaching_flag:
                # re-calculate co-teaching graph
                self._dynamic_params["tail_head_teachers"] = adj_func(info_dict)
                # when training plateaus, freeze the best model for the next epoch
                # if epoch_best_acc < prev_best_acc:
                #    self._dynamic_params["frozen"] = [epoch_best_idx]
                # else:
                #    self._dynamic_params["frozen"] = []
            else:
                # no coteaching
                trivial_adj = identity_adjacency(info_dict)
                self._dynamic_params["tail_head_teachers"] = trivial_adj
                self._dynamic_params["frozen"] = []
            prev_best_acc = max(prev_best_acc, epoch_best_acc)

        return train_info

    def adjust_optimizer_params(self):
        """
        ???+ note "Adjust all optimizer params."
        """
        for _net, _dict in zip(self.vector_nets, self._dynamic_params["optimizer"]):
            _net._dynamic_params["optimizer"] = _dict.copy()
            _net.adjust_optimizer_params()

    def train_epoch(self, train_loader):
        if self.verbose > 1:
            self._info(f"dynamic params {self._dynamic_params}")

        self.adjust_optimizer_params()

        for batch_idx, (loaded_input_list, loaded_output, _) in enumerate(train_loader):
            self._dynamic_params["batch"] = batch_idx + 1
            self.train_batch(loaded_input_list, loaded_output)

    def train_batch(self, loaded_input_list, loaded_output):
        """
        ???+ note "Train all neural networks for one batch."

            | Param               | Type           | Description             |
            | :------------------ | :------------- | :---------------------- |
            | `loaded_input_list` | `list` of `torch.Tensor` | input tensors |
            | `loaded_output`     | `torch.Tensor` | output tensor           |
            | `verbose`           | `int`          | verbosity for logging   |
        """
        denoise_rate = self._dynamic_params["denoise_rate"]
        tail_head_teachers = self._dynamic_params.get(
            "tail_head_teachers", [[i] for i, _net in enumerate(self.vector_nets)]
        )
        frozen_indices = self._dynamic_params.get("frozen", [])

        # determine which nets to enable train mode
        for i, _net in enumerate(self.vector_nets):
            if i not in frozen_indices:
                _net.nn.train()

        # compute logits
        output_tensor = loaded_output.float()
        net_inp_pairs = zip(self.vector_nets, loaded_input_list)
        logits_list = [_net.nn(_inp.float()) for _net, _inp in net_inp_pairs]

        loss_list = loss_coteaching_graph(
            logits_list,
            output_tensor,
            tail_head_teachers,
            denoise_rate,
        )

        for i, (_net, _loss) in enumerate(zip(self.vector_nets, loss_list)):
            if i not in frozen_indices:
                _net.nn_optimizer.zero_grad()
                _loss.backward()
                _net.nn_optimizer.step()

        if self.verbose > 0:
            log_info = dict(self._dynamic_params)
            log_info["performance"] = "".join(
                [
                    "|M{0}: L {1:.3f}".format(i, _loss)
                    for i, _loss in enumerate(loss_list)
                ]
            )
            print(
                "{0: <80}".format(
                    "Train: Epoch {epoch} Batch {batch} {performance}".format(
                        **log_info
                    )
                ),
                end="\r",
            )

    def evaluate_individual(self, dev_loader):
        """
        ???+ note "Evaluate each VectorNet against a dev set."

            | Param        | Type         | Description                |
            | :----------- | :----------- | :------------------------- |
            | `dev_loader` | `torch.utils.data.DataLoader` | dev set   |
        """
        for _net in self.vector_nets:
            _net.nn.eval()

        true = []
        pred_list = [[] for _net in self.vector_nets]
        for loaded_input_list, loaded_output, _idx in dev_loader:
            net_inp_pairs = zip(self.vector_nets, loaded_input_list)
            _output_tensor = loaded_output.float()
            _true_batch = _output_tensor.argmax(dim=1).detach().numpy()
            true.append(_true_batch)

            for i, (_net, _inp) in enumerate(net_inp_pairs):
                _logits = _net.nn(_inp.float())
                _probs = F.softmax(_logits, dim=1)
                _pred_batch = _probs.argmax(dim=1).detach().numpy()
                pred_list[i].append(_pred_batch)

        true = np.concatenate(true)
        pred_list = [np.concatenate(_pred) for _pred in pred_list]

        accuracy_list = [classification_accuracy(true, _pred) for _pred in pred_list]
        conf_mat_list = [confusion_matrix(true, _pred) for _pred in pred_list]
        disagree_rate = prediction_disagreement(pred_list, reduce=True)

        if self.verbose >= 0:
            log_info = dict(self._dynamic_params)
            log_info["performance"] = "".join(
                [
                    "|M{0}: Acc {1:.3f}".format(i, _acc)
                    for i, _acc in enumerate(accuracy_list)
                ]
            )
            self._info(
                "{0: <80}".format(
                    "Eval: Epoch {epoch} {performance}".format(**log_info)
                )
            )

        return accuracy_list, conf_mat_list, disagree_rate

    def evaluate_ensemble(self, dev_loader):
        """
        ???+ note "Evaluate against a dev set, adding up logits from all VectorNets."

            | Param        | Type         | Description                |
            | :----------- | :----------- | :------------------------- |
            | `dev_loader` | `torch.utils.data.DataLoader` | dev set   |
        """
        for _net in self.vector_nets:
            _net.nn.eval()

        true = []
        pred = []
        for loaded_input_list, loaded_output, _idx in dev_loader:
            net_inp_pairs = zip(self.vector_nets, loaded_input_list)
            _output_tensor = loaded_output.float()
            _true_batch = _output_tensor.argmax(dim=1).detach().numpy()
            true.append(_true_batch)

            _logits_sum = None
            for _net, _inp in net_inp_pairs:
                _logits = _net.nn(_inp.float()).detach()
                if _logits_sum is None:
                    _logits_sum = _logits.clone()
                else:
                    _logits_sum = _logits_sum.add(_logits.clone())

            _pred_batch = F.softmax(_logits_sum, dim=1).argmax(dim=1).detach().numpy()
            pred.append(_pred_batch)

        true = np.concatenate(true)
        pred = np.concatenate(pred)

        accuracy = classification_accuracy(true, pred)
        conf_mat = confusion_matrix(true, pred)

        if self.verbose >= 0:
            log_info = dict(self._dynamic_params)
            log_info["performance"] = "Ensemble Acc {0:.3f}".format(accuracy)
            self._info(
                "{0: <80}".format(
                    "Eval: Epoch {epoch} {performance}".format(**log_info)
                )
            )

        return accuracy, conf_mat
