from typing import Any, Protocol, Self

import torch

import ans


class ClassificationModel(Protocol):
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> tuple[float, torch.Tensor]:
        """
        Performs one step of training: forward pass, loss computation, backward pass, parameters update.

        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
            learning_rate: gradient descent step size
        Returns:
            loss: average loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ...

    def val_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
        """
        Performs one step of validation: forward pass, loss computation.

        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
            learning_rate: gradient descent step sizea
        Returns:
            loss: average loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ...


class LinearSoftmaxModel:
    def __init__(self, in_size: int, out_size: int, weight_scale: float = 1e-3) -> None:
        """
        Args:
            in_size: input dimension D
            out_size: number of classes K
            weight_scale: standard deviation of the normal distribution or a bound U(-weight_scale, weight_scale)
                          of uniform distribution used to initialize the weights
        """
        self.weight = torch.randn(in_size, out_size) * weight_scale
        self.bias = torch.zeros(out_size)

    def to(self, device: torch.device | str) -> Self:
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)
        return self

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 1e-4,
    ) -> tuple[float, torch.Tensor]:
        """
        Performs one step of training of linear softmax cross entropy classifier:
        - s = w * x + b
        - loss = cross_entropy(s, y)
        - compute gradients dloss/dw, dloss/db
        - params = params - learning_rate * dloss/dparams (SGD update)

        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
            learning_rate: gradient descent step size
        Returns:
            loss: average cross entropy loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        inputs = inputs.reshape(inputs.shape[0], -1)
        logits = inputs @ self.weight + self.bias
        smax = torch.max(logits, dim=-1, keepdim=True).values

        exp_args = torch.exp(logits - smax)
        sum_exp = torch.sum(exp_args, dim=-1, keepdim=True)
        probs = exp_args / sum_exp

        correct_scores = logits[torch.arange(inputs.shape[0]), targets]
        loss_per_sample = -correct_scores + smax.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
        data_loss = torch.mean(loss_per_sample)
        reg_loss = 0.5 * reg * torch.sum(self.weight ** 2)
        loss = data_loss + reg_loss

        N = inputs.shape[0]
        d_logits = probs.clone()
        d_logits[torch.arange(N), targets] -= 1
        d_logits /= N
        d_weight = inputs.T @ d_logits + reg * self.weight
        d_bias = torch.sum(d_logits, dim=0)

        self.weight -= learning_rate * d_weight
        self.bias -= learning_rate * d_bias

        return loss, logits

    def val_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
        """
        Performs one step of validation of linear softmax cross entropy classifier:
        - s = w * x + b
        - loss = cross_entropy(s, y)

        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
        Returns:
            loss: average cross entropy loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        inputs = inputs.reshape(inputs.shape[0], -1)
        logits = inputs @ self.weight + self.bias

        logits_max = torch.max(logits, dim=-1, keepdim=True).values
        exp_scores = torch.exp(logits - logits_max)
        sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)

        correct_scores = logits[torch.arange(inputs.shape[0]), targets]
        per_sample_loss = -correct_scores + logits_max.squeeze(-1) + torch.log(sum_exp_scores.squeeze(-1))
        loss = torch.mean(per_sample_loss)

        return loss, logits


class LinearSVMModel(LinearSoftmaxModel):
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 5e-4,
    ) -> tuple[float, torch.Tensor]:
        """
        Performs one step of training of linear support vector machine (SVM) classifier:
        - s = w * x + b
        - loss = hinge_loss(s, y)
        - compute gradients dloss/dw, dloss/db
        - params = params - learning_rate * dloss/dparams (SGD update)

        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
            learning_rate: gradient descent step size
        Returns:
            loss: average hinge loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        inputs = inputs.reshape(inputs.shape[0], -1)
        logits = inputs @ self.weight + self.bias
        N, K = logits.shape

        correct_scores = logits[torch.arange(N), targets].unsqueeze(1)
        margins = torch.clamp(logits - correct_scores + 1.0, min=0.0)
        margins[torch.arange(N), targets] = 0.0

        data_loss = torch.mean(torch.sum(margins, dim=1))
        reg_loss = 0.5 * reg * torch.sum(self.weight ** 2)
        loss = data_loss + reg_loss

        mask = (margins > 0.0).float()
        row_sum_mask = torch.sum(mask, dim=1)
        mask[torch.arange(N), targets] = -row_sum_mask

        d_logits = mask / N
        d_weight = inputs.T @ d_logits + reg * self.weight
        d_bias = torch.sum(d_logits, dim=0)

        self.weight -= learning_rate * d_weight
        self.bias -= learning_rate * d_bias

        return loss, logits

    def val_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
        """
        Performs one step of validation of linear support vector machine (SVM) classifier:
        - s = w * x + b
        - loss = hinge_loss(s, y)

        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
        Returns:
            loss: average hinge loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        inputs = inputs.reshape(inputs.shape[0], -1)
        logits = inputs @ self.weight + self.bias
        N, K = logits.shape

        correct_scores = logits[torch.arange(N), targets].unsqueeze(1)
        margins = torch.clamp(logits - correct_scores + 1.0, min=0.0)
        margins[torch.arange(N), targets] = 0.0

        loss = torch.mean(torch.sum(margins, dim=1))

        return loss, logits


class TwoLayerPerceptron:
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        out_size: int,
        weight_scale: float = 0.001,
    ) -> None:
        ########################################
        # TODO: implement

        self.weight1 = ...
        self.bias1 = ...
        self.weight2 = ...
        self.bias2 = ...

        # ENDTODO
        ########################################

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3,
    ) -> tuple[float, torch.Tensor]:
        ########################################
        # TODO: implement
        #
        # Some of the intermediate gradients that are to be expected for the test data used in
        # test_two_layer_perceptron.TestTrainStep:
        #   dlogits = torch.tensor([  # gradient w.r.t. output of the network
        #     [ 0.0072,  0.0924, -0.0996],
        #     [-0.0484,  0.0199,  0.0285],
        #     [ 0.0047, -0.0258,  0.0211],
        #     [ 0.0126,  0.0778, -0.0904],
        #     [ 0.0333, -0.0696,  0.0362],
        #     [ 0.0696,  0.0257, -0.0953],
        #     [-0.0290,  0.0066,  0.0223],
        #     [ 0.1003,  0.0059, -0.1062]
        #   ])
        #   dhidden = torch.tensor([  # gradient w.r.t. output of sigmoid
        #     [ 0.1521,  0.0635, -0.1338, -0.0515],
        #     [ 0.0342,  0.0584, -0.1164,  0.0261],
        #     [-0.0427, -0.0237,  0.0491,  0.0094],
        #     [ 0.1279,  0.0477, -0.1013, -0.0482],
        #     [-0.1156, -0.0825,  0.1688,  0.0099],
        #     [ 0.0404, -0.0429,  0.0813, -0.0646],
        #     [ 0.0118,  0.0310, -0.0613,  0.0182],
        #     [ 0.0070, -0.0853,  0.1665, -0.0776]
        #   ])
        #   dprehidden = torch.tensor([  # gradient w.r.t. output of the first linear layer (before sigmoid)
        #     [ 0.0338,  0.0038, -0.0153, -0.0002],
        #     [ 0.0047,  0.0002, -0.0291,  0.0004],
        #     [-0.0092, -0.0003,  0.0007,  0.0001],
        #     [ 0.0315,  0.0097, -0.0107, -0.0039],
        #     [-0.0284, -0.0081,  0.0328,  0.0020],
        #     [ 0.0091, -0.0002,  0.0203, -0.0102],
        #     [ 0.0003,  0.0007, -0.0072,  0.0028],
        #     [ 0.0003, -0.0001,  0.0153, -0.0004]])
        #   ])

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss, logits

    def val_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
        """
        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
        Returns:
            loss: average loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss, logits

    def save(self, filename: str) -> None:
        torch.save(dict(weight1=self.weight1, bias1=self.bias1, weight2=self.weight2, bias2=self.bias2), filename)

    @classmethod
    def load(cls, filename: str) -> Self:
        dic = torch.load(filename, weights_only=True)
        model = cls(*dic["weight1"].shape, dic["weight2"].shape[1])
        model.weight1 = dic["weight1"]
        model.bias1 = dic["bias1"]
        model.weight2 = dic["weight2"]
        model.bias2 = dic["bias2"]
        return model


def accuracy(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Args:
        scores: output linear scores (logits or probabilities); shape (num_samples, num_classes)
        targets: vector of class indicies (integers); shape (num_samples,)
    Returns:
        acc: average accuracy on the batch; single number (scalar), e.g. 0.364
    """
    predicted = torch.argmax(scores, dim=-1)
    correct = (predicted == targets)
    return correct.float().mean().item()


def train_epoch(model: ClassificationModel, loader: ans.data.BatchLoader, **train_step_kwargs) -> Any:
    """
    Trains `model` on the dataset `loader` for one epoch.

    Args:
        model: Model to train. Must have method `train_step`.
        loader: loader of the training dataset
    Returns:
        return whatever is needed
    """
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for inputs, targets in loader:
        loss, logits = model.train_step(inputs, targets, **train_step_kwargs)
        acc = accuracy(logits, targets)

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches

    return avg_loss, avg_acc


def validate(model: ClassificationModel, loader: ans.data.BatchLoader, **val_step_kwargs) -> tuple[float, float]:
    """
    Validates `model` on the dataset `loader`.

    Args:
        model: Model to be validated. Must have method `val_step`.
        loader: loader of the training dataset
    Returns:
        mean_loss: average loss achieved on the dataset during training
        mean_acc: average accuracy achieved on the dataset during model training
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        batch_size = inputs.shape[0]
        loss, logits = model.val_step(inputs, targets, **val_step_kwargs)

        predicted = torch.argmax(logits, dim=-1)
        correct = (predicted == targets).sum().item()

        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc