"""
???+ note "Neural network components."

    `torch`-based template classes for implementing neural nets that work the most smoothly with `hover`
"""
import torch
import torch.nn.functional as F
from hover.core import Loggable
from hover.utils.metrics import classification_accuracy
from hover.utils.misc import current_time
from hover.utils.torch_helper import cross_entropy_with_probs
from hover.utils.denoising import (
    loss_coteaching_graph,
    prediction_disagreement,
    disagreement_priority,
)
from sklearn.metrics import confusion_matrix
import numpy as np


class VectorNet(Loggable):

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
        """

        assert isinstance(
            verbose, int
        ), f"Expected verbose as int, got {type(verbose)} {verbose}"
        self.verbose = verbose

        # set up label conversion
        self.label_encoder = {_label: i for i, _label in enumerate(labels)}
        self.label_decoder = {i: _label for i, _label in enumerate(labels)}
        self.num_classes = len(self.label_encoder)

        # set up vectorizer and the neural network with appropriate dimensions
        self.vectorizer = vectorizer
        vec_dim = self.vectorizer("").shape[0]
        self.nn = architecture(vec_dim, self.num_classes)

        # if a state dict exists, load it and create a backup copy
        import os

        if os.path.isfile(state_dict_path):
            from shutil import copyfile

            try:
                self.nn.load_state_dict(torch.load(state_dict_path))
            except Exception as e:
                self._warn(f"Load VectorNet state path failed with {type(e)}: e")

            if backup_state_dict:
                state_dict_backup_path = (
                    f"{state_dict_path}.{current_time('%Y%m%d%H%M%S')}"
                )
                copyfile(state_dict_path, state_dict_backup_path)

        # set a path to store updated parameters
        self.nn_update_path = state_dict_path

        # initialize an optimizer object and a dict to hold dynamic parameters
        optimizer_cls = optimizer_cls or self.__class__.DEFAULT_OPTIM_CLS
        optimizer_kwargs = (
            optimizer_kwargs or self.__class__.DEFAULT_OPTIM_KWARGS.copy()
        )
        self.nn_optimizer = optimizer_cls(self.nn.parameters())
        assert isinstance(
            self.nn_optimizer, torch.optim.Optimizer
        ), f"Expected an optimizer, got {type(self.nn_optimizer)}"
        self._dynamic_params = {"optimizer": optimizer_kwargs}

    @classmethod
    def from_module(cls, model_module, labels):
        """
        ???+ note "Create a VectorNet model from a loadable module."

            | Param          | Type       | Description                          |
            | :------------- | :--------- | :----------------------------------- |
            | `model_module` | `module` or `str` | (path to) a local Python workspace module which contains a get_vectorizer() callable, get_architecture() callable, and a get_state_dict_path() callable |
            | `labels`       | `list`     | list of `str` classification labels  |
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
        )

        return model

    def save(self, save_path=None):
        """
        ???+ note "Save the current state dict with authorization to overwrite."
            | Param       | Type  | Description                           |
            | :---------- | :---- | :------------------------------------ |
            | `save_path` | `str` | option alternative path to state dict |
        """
        if save_path is None:
            save_path = self.nn_update_path
        torch.save(self.nn.state_dict(), save_path)

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

    def train(self, train_loader, dev_loader=None, epochs=1):
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

        if self.verbose > 0:
            log_info = dict(self._dynamic_params)
            log_info["performance"] = "Acc {0:.3f}".format(accuracy)
            self._info(
                "{0: <80}".format(
                    "Eval: Epoch {epoch} {performance}".format(**log_info)
                )
            )

        return accuracy, conf_mat


class MultiVectorNet(Loggable):

    """
    ???+ note "Ensemble transfer learning model: multiple jointly-trained VectorNet's."

        Note that the VectorNets can have different vectorizers.
        Consequently, when training the nets, they expect multiple vectors per input.

        Coupled with:

        -   `hover.utils.torch_helper.MultiVectorDataset`
    """

    DEFAULT_ADJACENCY_FUNC = disagreement_priority

    def __init__(self, vector_nets, verbose=0):
        """
        ???+ note "Create the `VectorNet`, loading parameters if available."
            | Param           | Type   | Description                  |
            | :-------------- | :----- | :--------------------------- |
            | `vector_nets`   | `list` | list of VectorNet instances  |
            | `verbose`       | `int`  | logging verbosity level      |
        """
        self.vector_nets = vector_nets
        self._dynamic_params = dict()
        assert isinstance(
            verbose, int
        ), f"Expected verbose as int, got {type(verbose)} {verbose}"
        self.verbose = verbose
        self._warn(
            "this class is in preview and is not sufficiently tested. Use with caution."
        )

    def train(self, train_loader, params_per_epoch, dev_loader=None):
        """
        ???+ note "Train multiple VectorNet's jointly."

            | Param          | Type         | Description                |
            | :------------- | :----------- | :------------------------- |
            | `train_loader` | `torch.utils.data.DataLoader` | train set |
            | `params_per_epoch` | `list` of `dict` | updates to dynamic params |
            | `dev_loader`   | `torch.utils.data.DataLoader` | dev set   |

            Example for params_per_epoch:

            ```
            def get_params(warmup_epochs=5, coteach_epochs=10, forget_rate=0.3):
                for i in range(warmup_epochs):
                    yield {"forget_rate": 0.0, "optimizer": [{"lr": 0.1, "momentum": 0.9}] * 4}
                for i in range(coteach_epochs):
                    yield {"forget_rate": forget_rate, "optimizer": [{"lr": 0.05, "momentum": 0.7}] * 4}
            params_per_epoch = get_params()
            ```
        """
        train_info = []
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
            self._dynamic_params["tail_head_teachers"] = adj_func(info_dict)

        return train_info

    def adjust_optimizer_params(self):
        """
        ???+ note "Adjust all optimizer params."
        """
        for _net, _dict in zip(self.vector_nets, self._dynamic_params["optimizer"]):
            _net._dynamic_params["optimizer"] = _dict.copy()
            _net.adjust_optimizer_params()

    def train_epoch(self, train_loader):
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
        forget_rate = self._dynamic_params["forget_rate"]
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
            forget_rate,
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

        if self.verbose > 0:
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

        if self.verbose > 0:
            log_info = dict(self._dynamic_params)
            log_info["performance"] = "Ensemble Acc {0:.3f}".format(accuracy)
            self._info(
                "{0: <80}".format(
                    "Eval: Epoch {epoch} {performance}".format(**log_info)
                )
            )

        return accuracy, conf_mat
