"""
???+ note "Neural network components."

    `torch`-based template classes for implementing neural nets that work the most smoothly with `hover`
"""
import torch
import torch.nn.functional as F
from deprecated import deprecated
from hover.core import Loggable
from hover.utils.metrics import classification_accuracy
from hover.utils.misc import current_time
from sklearn.metrics import confusion_matrix
from snorkel.classification import cross_entropy_with_probs
import numpy as np


@deprecated(
    version="0.4.0",
    reason="will be removed in a future version (planned 0.5.0); please use VectorNet.from_module() instead.",
)
def create_vector_net_from_module(specific_class, model_module_name, labels):
    """
    ???+ warning "Deprecated into a trivial invocation of VectorNet's class method."
    """
    return specific_class.from_module(model_module_name, labels)


class VectorNet(Loggable):

    """
    ???+ note "Simple transfer learning model: a user-supplied vectorizer followed by a neural net."
        This is a parent class whose children may use different training schemes.

        Coupled with:

        -   `hover.utils.torch_helper.VectorDataset`
        -   `hover.utils.torch_helper.vector_dataloader`
    """

    def __init__(
        self, vectorizer, architecture, state_dict_path, labels, backup_state_dict=True
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
        """

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
        self.nn_optimizer = torch.optim.Adam(self.nn.parameters())
        self._dynamic_params = {"optimizer": {"lr": 0.01, "betas": (0.9, 0.999)}}

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
        ???+ note "**UPCOMING**"
            Compute a trajectory of manifold propagation through the neural net.

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

    def evaluate(self, dev_loader, verbose=1):
        """
        ???+ note "Evaluate the VecNet against a dev set."

            | Param        | Type         | Description                |
            | :----------- | :----------- | :------------------------- |
            | `dev_loader` | `torch.utils.data.DataLoader` | dev set   |
            | `verbose`    | `int`        | verbosity for logging      |
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

        if verbose > 0:
            log_info = dict(self._dynamic_params)
            log_info["performance"] = "Acc {0:.3f}".format(accuracy)
            self._info(
                "{0: <80}".format(
                    "Eval: Epoch {epoch} {performance}".format(**log_info)
                )
            )

        return accuracy, conf_mat

    def train(self, train_loader, dev_loader=None, epochs=1, verbose=1):
        """
        ???+ note "Train the neural network part of the VecNet."

            - This method is a vanilla template and is intended to be overridden in child classes.
            - Also intended to be coupled with self.train_batch().

            | Param          | Type         | Description                |
            | :------------- | :----------- | :------------------------- |
            | `train_loader` | `torch.utils.data.DataLoader` | train set |
            | `dev_loader`   | `torch.utils.data.DataLoader` | dev set   |
            | `epochs`       | `int`        | number of epochs to train  |
            | `verbose`      | `int`        | verbosity for logging      |
        """
        train_info = []
        for epoch_idx in range(epochs):
            self._dynamic_params["epoch"] = epoch_idx + 1
            self.train_epoch(train_loader, verbose=verbose)
            if dev_loader is not None:
                acc, conf_mat = self.evaluate(dev_loader, verbose=verbose)
                train_info.append({"accuracy": acc, "confusion_matrix": conf_mat})
        return train_info

    def train_epoch(self, train_loader, *args, **kwargs):
        """
        ???+ note "Train the neural network for one epoch."

            - Supports flexible args and kwargs for child classes that may implement self.train() and self.train_batch() differently.

            | Param          | Type         | Description                |
            | :------------- | :----------- | :------------------------- |
            | `train_loader` | `torch.utils.data.DataLoader` | train set |
            | `dev_loader`   | `torch.utils.data.DataLoader` | dev set   |
            | `verbose`      | `int`        | verbosity for logging      |
            | `*args`        | | arguments to forward to `train_batch`   |
            | `**kwargs`     | | kwargs to forward to `train_batch`      |
        """
        self.adjust_optimizer_params()
        for batch_idx, (loaded_input, loaded_output, _) in enumerate(train_loader):
            self._dynamic_params["batch"] = batch_idx + 1
            self.train_batch(loaded_input, loaded_output, *args, **kwargs)

    def train_batch(self, loaded_input, loaded_output, verbose=1):
        """
        ???+ note "Train the neural network for one batch."

            | Param           | Type           | Description           |
            | :-------------- | :------------- | :-------------------- |
            | `loaded_input`  | `torch.Tensor` | input tensor          |
            | `loaded_output` | `torch.Tensor` | output tensor         |
            | `verbose`       | `int`          | verbosity for logging |
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

        if verbose > 0:
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
