import torch
import torch.nn.functional as F
from datetime import datetime
from hover.utils.metrics import classification_accuracy
from wasabi import msg as logger
from sklearn.metrics import confusion_matrix
from snorkel.classification import cross_entropy_with_probs
import numpy as np


def create_vector_net_from_module(specific_class, model_module_name, labels):
    """
    Create a TextVectorNet model, or of its child class.

    - param specific_class(class): TextVectorNet or its child class.
    - param model_module_name(str): path to a local Python module in the working directory whose __init__.py file contains a get_vectorizer() callable, get_architecture() callable, and a get_state_dict_path() callable.
    - param labels(list of str): the classification labels, e.g. ["POSITIVE", "NEGATIVE"].
    """
    from importlib import import_module

    model_module = import_module(model_module_name)

    # Load the model by retrieving the inp-to-vec function, architecture, and state dict
    model = specific_class(
        model_module.get_vectorizer(),
        model_module.get_architecture(),
        model_module.get_state_dict_path(),
        labels,
    )

    return model


class VectorNet(object):

    """
    Simple transfer learning model: a user-supplied vectorizer followed by a neural net.

    This is a parent class whose children may use different training schemes.

    Please refer to hover.utils.torch_helper.VectorDataset and vector_dataloader for more info.
    """

    def __init__(self, vectorizer, architecture, state_dict_path, labels):
        """
        - param vectorizer(callable): a function that converts any string to a NumPy 1-D array.
        - param architecture(class): a `torch.nn.Module` child class to be instantiated into a neural net.
        - param state_dict_path(str): path to a PyTorch state dict that matches the architecture.
        - param labels(list of str): the classification labels, e.g. ["POSITIVE", "NEGATIVE"].
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
                logger.warn(f"Load VectorNet state path failed with {type(e)}: e")

            state_dict_backup_path = (
                f"{state_dict_path}.{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )
            copyfile(state_dict_path, state_dict_backup_path)

        # set a path to store updated parameters
        self.nn_update_path = state_dict_path

        # initialize an optimizer object and a dict to hold dynamic parameters
        self.nn_optimizer = torch.optim.Adam(self.nn.parameters())
        self._dynamic_params = {"optimizer": {"lr": 0.01, "betas": (0.9, 0.999)}}

    def save(self, save_path=None):
        """
        Save the current state dict with authorization to overwrite.
        """
        if save_path is None:
            save_path = self.nn_update_path
        torch.save(self.nn.state_dict(), save_path)

    def adjust_optimizer_params(self):
        """
        Dynamically change parameters of the neural net optimizer.

        - Intended to be polymorphic in child classes and to be called per epoch.
        """
        for _group in self.nn_optimizer.param_groups:
            _group.update(self._dynamic_params["optimizer"])

    def predict_proba(self, inps):
        """
        End-to-end single/multi-piece prediction from inp to class probabilities.
        """
        # if the input is a single piece of inp, cast it to a list
        FLAG_SINGLE = isinstance(inps, str)
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
        TODO: need a clean way to pass kwargs to dimensionality reduction.

        1. vectorize inps
        2. forward propagate, keeping intermediates
        3. fit intermediates to 2D manifolds
        4. fit manifolds using Procrustes shape analysis
        5. fit shapes to trajectory splines

        - param inps(list): input to calculate the manifold profile from.
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
        Evaluate the neural network against a dev set.
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
            logger.info(
                "{0: <80}".format(
                    "Eval: Epoch {epoch} {performance}".format(**log_info)
                )
            )

        return accuracy, conf_mat

    def train(self, train_loader, dev_loader=None, epochs=1, verbose=1):
        """
        Train the neural network.

        - This method is a vanilla template and is intended to be overridden in child classes.
        - Also intended to be coupled with self.train_batch().
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
        Train the neural network for one epoch.

        - Supports flexible args and kwargs for child classes that may implement self.train() and self.train_batch() differently.
        """
        self.adjust_optimizer_params()
        for batch_idx, (loaded_input, loaded_output, _) in enumerate(train_loader):
            self._dynamic_params["batch"] = batch_idx + 1
            self.train_batch(loaded_input, loaded_output, *args, **kwargs)

    def train_batch(self, loaded_input, loaded_output, verbose=1):
        """
        Train the neural network for one batch.
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
            print(
                "{0: <80}".format(
                    "Train: Epoch {epoch} Batch {batch} {performance}".format(
                        **log_info
                    )
                ),
                end="\r",
            )
