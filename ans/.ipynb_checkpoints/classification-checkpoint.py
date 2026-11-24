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
            learning_rate: gradient descent step size
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
        ########################################
        self.weight = torch.randn(in_size, out_size) * weight_scale
        self.bias = torch.zeros(out_size)
        # ENDTODO
        ########################################

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
        ########################################
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
        # ENDTODO
        ########################################

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
        ########################################
        inputs = inputs.reshape(inputs.shape[0], -1)
        logits = inputs @ self.weight + self.bias

        logits_max = torch.max(logits, dim=-1, keepdim=True).values
        exp_scores = torch.exp(logits - logits_max)
        sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)

        correct_scores = logits[torch.arange(inputs.shape[0]), targets]
        per_sample_loss = -correct_scores + logits_max.squeeze(-1) + torch.log(sum_exp_scores.squeeze(-1))
        loss = torch.mean(per_sample_loss)
        # ENDTODO
        ########################################

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
        ########################################
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
        # ENDTODO
        ########################################

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
        ########################################
        inputs = inputs.reshape(inputs.shape[0], -1)
        logits = inputs @ self.weight + self.bias
        N, K = logits.shape

        correct_scores = logits[torch.arange(N), targets].unsqueeze(1)
        margins = torch.clamp(logits - correct_scores + 1.0, min=0.0)
        margins[torch.arange(N), targets] = 0.0

        loss = torch.mean(torch.sum(margins, dim=1))
        # ENDTODO
        ########################################

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
        self.weight1 = torch.randn(in_size, hidden_size) * weight_scale
        self.bias1 = torch.zeros(hidden_size)
        self.weight2 = torch.randn(hidden_size, out_size) * weight_scale
        self.bias2 = torch.zeros(out_size)
        # ENDTODO
        ########################################

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3,
    ) -> tuple[float, torch.Tensor]:
        ########################################
        r = inputs @ self.weight1 + self.bias1
        h = torch.sigmoid(r)
        s = h @ self.weight2 + self.bias2

        snExp = torch.exp(s)
        sumSnExp = torch.sum(snExp, dim=1, keepdim=True)
        softmaxProbs = snExp / sumSnExp

        softmaxProbsSnYn = softmaxProbs[torch.arange(inputs.shape[0]), targets]
        loss = -torch.log(softmaxProbsSnYn).mean()
        logits = s

        batch_size = inputs.shape[0]
        one_hot = torch.nn.functional.one_hot(targets, num_classes=s.shape[1]).float()

        dlogits = (softmaxProbs - one_hot) / batch_size
        dhidden = dlogits @ self.weight2.T
        dprehidden = dhidden * h * (1 - h)

        dweight2 = h.T @ dlogits
        dbias2 = dlogits.sum(dim=0)

        dweight1 = inputs.T @ dprehidden
        dbias1 = dprehidden.sum(dim=0)

        with torch.no_grad():
            self.weight1 -= learning_rate * dweight1
            self.bias1 -= learning_rate * dbias1
            self.weight2 -= learning_rate * dweight2
            self.bias2 -= learning_rate * dbias2
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
        r = inputs @ self.weight1 + self.bias1
        h = torch.sigmoid(r)
        s = h @ self.weight2 + self.bias2

        snExp = torch.exp(s)
        sumSnExp = torch.sum(snExp, dim=1, keepdim=True)
        softmaxProbs = snExp / sumSnExp

        softmaxProbsSnYn = softmaxProbs[torch.arange(inputs.shape[0]), targets]
        loss = -torch.log(softmaxProbsSnYn).mean()
        logits = s
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


class TwoLayerPerceptronAutograd:
    def __init__(self, in_size: int, hidden_size: int, out_size: int, weight_scale: float = 0.001) -> None:
        self.weight1 = ans.autograd.Variable(torch.randn(in_size, hidden_size) * weight_scale)
        self.bias1 = ans.autograd.Variable(torch.zeros(hidden_size))
        self.weight2 = ans.autograd.Variable(torch.randn(hidden_size, out_size) * weight_scale)
        self.bias2 = ans.autograd.Variable(torch.zeros(out_size))

    def _forward(self, inputs: torch.Tensor) -> ans.autograd.Variable:
        x = ans.autograd.Variable(inputs.reshape(inputs.shape[0], -1))
        hidden = (x @ self.weight1 + self.bias1).sigmoid()
        logits = hidden @ self.weight2 + self.bias2
        return logits

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3,
    ) -> tuple[float, torch.Tensor]:
        logits = self._forward(inputs)
        loss = softmax_cross_entropy(logits, targets)

        self.weight1.grad = None
        self.bias1.grad = None
        self.weight2.grad = None
        self.bias2.grad = None

        loss.backprop()

        if self.weight1.grad is not None:
            self.weight1.data = self.weight1.data - learning_rate * self.weight1.grad
        if self.bias1.grad is not None:
            self.bias1.data = self.bias1.data - learning_rate * self.bias1.grad
        if self.weight2.grad is not None:
            self.weight2.data = self.weight2.data - learning_rate * self.weight2.grad
        if self.bias2.grad is not None:
            self.bias2.data = self.bias2.data - learning_rate * self.bias2.grad

        return loss.data.item(), logits.data

    def val_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
        """
        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
        Returns:
            loss: average loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        logits = self._forward(inputs)
        loss = softmax_cross_entropy(logits, targets)
        # ENDTODO
        ########################################

        return loss.data.item(), logits.data

    def save(self, filename: str) -> None:
        torch.save(
            dict(weight1=self.weight1.data, bias1=self.bias1.data, weight2=self.weight2.data, bias2=self.bias2.data),
            filename,
        )

    @classmethod
    def load(cls, filename: str) -> Self:
        dic = torch.load(filename, weights_only=True)
        model = cls(*dic["weight1"].shape, dic["weight2"].shape[1])
        model.weight1 = ans.autograd.Variable(dic["weight1"])
        model.bias1 = ans.autograd.Variable(dic["bias1"])
        model.weight2 = ans.autograd.Variable(dic["weight2"])
        model.bias2 = ans.autograd.Variable(dic["bias2"])
        return model


class AutogradClassifier(ans.nn.Module):
    def __init__(self, backbone: ans.nn.Module, optimizer: ans.nn.Optimizer) -> None:
        self.backbone = backbone
        self.optimizer = optimizer

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3,
    ) -> tuple[float, torch.Tensor]:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss.data.item(), logits.data

    def val_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[float, torch.Tensor]:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss.data.item(), logits.data

    def save(self, filename: str) -> None:
        torch.save(self, filename)  # not the correct & safe way to do this

    @classmethod
    def load(cls, filename: str) -> Self:
        return torch.load(filename, weights_only=False)  # not the correct & safe way to do this


def accuracy(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Args:
        scores: output linear scores (logits or probabilities); shape (num_samples, num_classes)
        targets: vector of class indicies (integers); shape (num_samples,)
    Returns:
        acc: average accuracy on the batch; single number (scalar), e.g. 0.364
    """
    ########################################
    predicted = torch.argmax(scores, dim=-1)
    correct = predicted == targets
    return correct.float().mean().item()
    # ENDTODO
    ########################################


def softmax_cross_entropy(
    logits: torch.Tensor | ans.autograd.Variable,
    targets: torch.Tensor,
) -> torch.Tensor | ans.autograd.Variable:
    """
    Args:
        logits: output linear scores (logits); shape (num_samples, num_classes)
        targets: vector of class indicies (integers); shape (num_samples,)
    Returns:
        loss: average loss on the batch; single number (scalar), e.g. 2.31
    """
    ########################################
    n_samples = logits.shape[0] if isinstance(logits, torch.Tensor) else logits.data.shape[0]
    row_idx = torch.arange(n_samples)

    if isinstance(logits, ans.autograd.Variable):
        exp_scores = logits.exp()
        sum_exp = exp_scores.sum(dim=-1, keepdim=True)
        probs = exp_scores / sum_exp
        target_probs = probs[row_idx, targets]
        loss = (target_probs.log() * (-1.0)).sum() / n_samples
        return loss

    # Varianta pro torch.Tensor (numerický stabilní softmax cross-entropy)
    max_scores = torch.max(logits, dim=-1, keepdim=True).values
    shifted = logits - max_scores
    exp_scores = torch.exp(shifted)
    sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True)
    log_probs = shifted - torch.log(sum_exp)
    loss = -log_probs[row_idx, targets].mean()
    return loss
    # ENDTODO
    ########################################


def train_epoch(model: ClassificationModel, loader: ans.data.BatchLoader, **train_step_kwargs) -> Any:
    """
    Trains `model` on the dataset `loader` for one epoch.

    Args:
        model: Model to train. Must have method `train_step`.
        loader: loader of the training dataset
    Returns:
        return whatever is needed
    """
    ########################################
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for inputs, targets in loader:
        loss, logits = model.train_step(inputs, targets, **train_step_kwargs)
        acc = accuracy(logits, targets)

        total_loss += loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        total_acc += acc
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches

    return avg_loss, avg_acc
    # ENDTODO
    ########################################


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
    ########################################
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        batch_size = inputs.shape[0]
        loss, logits = model.val_step(inputs, targets, **val_step_kwargs)

        predicted = torch.argmax(logits, dim=-1)
        correct = (predicted == targets).sum().item()

        total_loss += (loss.item() if isinstance(loss, torch.Tensor) else float(loss)) * batch_size
        total_correct += correct
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc
    # ENDTODO
    ########################################