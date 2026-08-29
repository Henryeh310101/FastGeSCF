import numpy as np
import torch
import torchvision.transforms.functional as tvff
from torch.nn import functional as F
from torchvision import transforms as tvf

import py_utils.evaluation as evaluation

# local module
from project_config import resolve_device
from py_utils.src import utils, utils_torch, utils_img


def translate_image(tx, ty):

    def f(img):
        orig_shape = img.shape
        if len(orig_shape) == 2:
            # affine function cannot accept 2D image or an error will be raised
            # RuntimeError: grid_sampler(): expected grid ...
            # thus, change shape from (m, n) to (1, m, n)
            img = img.unsqueeze(0)

        x = tvf.functional.affine(
            img, angle=0, translate=(tx, ty), scale=1.0, shear=0
        )

        if len(orig_shape) == 2:
            x = x.squeeze(0)

        return x

    return f


def rotate_image(angle):

    def f(img):
        orig_shape = img.shape
        if len(orig_shape) == 2:
            # affine function cannot accept 2D image or an error will be raised
            # RuntimeError: grid_sampler(): expected grid ...
            # thus, change shape from (m, n) to (1, m, n)
            img = img.unsqueeze(0)

        x = tvf.functional.affine(
            img, angle=angle, translate=(0, 0), scale=1.0, shear=0
        )

        if len(orig_shape) == 2:
            x = x.squeeze(0)

        return x

    return f


class CDDataWrapper:

    def __init__(
        self,
        dataset,
        transform=None,
        target_transform=None,
        return_ind=False,
        translate0=(0, 0),
        translate1=(0, 0),
        rotate_angle0=0.0,
        rotate_angle1=0.0,
        hflip_prob=0.0,
        augment_diff_degree=None,
        augment_diff_translate=None,
    ):

        if transform is None:

            def transform(x):
                return x

        if target_transform is None:

            def target_transform(x):
                return x

        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.return_ind = return_ind
        self.hflip_prob = hflip_prob

        self.translate0 = translate_image(*translate0)
        self.translate1 = translate_image(*translate1)
        self.rotate0 = rotate_image(rotate_angle0)
        self.rotate1 = rotate_image(rotate_angle1)
        self._pre_transform = tvf.ToTensor()
        self._pos_transform = tvf.ToPILImage()

        if augment_diff_degree is None:
            augment_diff_degree = 0.0

        self.augment_diff_degree = np.abs(augment_diff_degree)
        self.augment_diff_translate = augment_diff_translate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        t0, t1, gt = self.dataset[idx]

        t0 = self._pre_transform(t0)
        t1 = self._pre_transform(t1)
        gt = self._pre_transform(gt)

        t0 = self.translate0(t0)
        t1 = self.translate1(t1)
        gt = self.translate0(gt)

        t0 = self.rotate0(t0)
        t1 = self.rotate1(t1)
        gt = self.rotate0(gt)

        t0 = self._pos_transform(t0)
        t1 = self._pos_transform(t1)
        gt = self._pos_transform(gt)

        t0 = self.transform(t0)
        t1 = self.transform(t1)
        gt = self.target_transform(gt)

        if self.augment_diff_degree > 0.0:

            degree = np.random.uniform(
                -self.augment_diff_degree,
                self.augment_diff_degree,
            )

            t0 = rotate_image(degree)(t0)
            gt = rotate_image(degree)(gt)

        if self.augment_diff_translate is not None:

            translate = np.random.uniform(
                self.augment_diff_translate[0],
                self.augment_diff_translate[1],
                size=2,
            )

            t0 = translate_image(*translate)(t0)
            gt = translate_image(*translate)(gt)

        if np.random.random() < self.hflip_prob:
            t0 = tvff.hflip(t0)
            t1 = tvff.hflip(t1)
            gt = tvff.hflip(gt)

        output = t0, t1, gt

        if self.return_ind:
            return idx, output

        return output


def _yield_CD_evaluation_after_cropping(model, dataset, crop_shape, device=None):

    device = resolve_device(device)

    for input_1, input_2, target in dataset:

        input_1 = input_1.to(device)
        input_2 = input_2.to(device)

        with torch.no_grad():
            predict = model(input_1, input_2)  # (batch, m, n, 2)
            predict = torch.argmax(predict, dim=-1)  # (batch, m, n)

        target = target.detach().cpu().numpy()  # (batch, m, n)
        predict = predict.detach().cpu().numpy()  # (batch, m, n)

        assert len(target.shape) == 3 and len(predict.shape) == 3
        assert target.shape == predict.shape

        # ravel out each item in a batch
        for pre, tar in zip(predict, target):

            tar = utils_img.center_crop_image(tar, crop_shape)
            pre = utils_img.center_crop_image(pre, crop_shape)

            R = evaluation.change_mask_metric(pre, tar)
            yield R["precision"], R["recall"], R["accuracy"], R["f1_score"]


def _yield_CD_evaluation(model, dataset, device=None):
    device = resolve_device(device)

    for input_1, input_2, target in dataset:

        input_1 = input_1.to(device)
        input_2 = input_2.to(device)

        with torch.no_grad():
            predict = model(input_1, input_2)  # (batch, m, n, 2)
            predict = torch.argmax(predict, dim=-1)  # (batch, m, n)

        target = target.detach().cpu().numpy()  # (batch, m, n)
        predict = predict.detach().cpu().numpy()  # (batch, m, n)

        assert len(target.shape) == 3 and len(predict.shape) == 3
        assert target.shape == predict.shape

        # ravel out each item in a batch
        for pre, tar in zip(predict, target):

            R = evaluation.change_mask_metric(pre, tar)
            yield R["precision"], R["recall"], R["accuracy"], R["f1_score"]


def image_change_detection_evaluation(
    model,
    data_loader,
    evaluator=None,
    prefix="",
    device="auto",
    verbose=True,
    dry_run=False,
    return_details=False,
    return_duration=False,
):
    """
    Evaluate the performance of a change detection model on a given dataset.

    Args:
        model (torch.nn.Module):
            The change detection model to be evaluated.
        data_loader (torch.utils.data.DataLoader):
            DataLoader for the evaluation dataset.
        evaluator (generator, optional):
            Custom evaluator yielding evaluation metrics.
            If None, a generator is formed by both `model` and `data_loader`.
            If not None, `model` and `data_loader` are ignored.
        prefix (str, optional):
            Prefix for progress messages. Default is an empty string.
        device (str, optional):
            The device to run the evaluation on ('cuda' or 'cpu').
            Default is 'cuda'.
        verbose (bool, optional):
            If True, print progress information during evaluation.
            Default is True.
        dry_run (bool, optional):
            If True, perform only one iteration for testing purposes.
            Default is False.
        return_details (bool, optional):
            If True, include detailed metrics for each iteration in the output.
            Default is False.
        return_duration (bool, optional):
            If True, include the total duration of the evaluation in the output.
            Default is False.

    Returns:
        tuple: A tuple containing:
            - dict: Aggregate evaluation metrics with keys
                'precision', 'recall', 'accuracy', and 'f1_score'.
            - float (optional): Total duration of the evaluation in seconds,
                if return_duration is True.
            - dict (optional): Detailed metrics for each iteration,
                if return_details is True.
    """

    precisions = []
    recalls = []
    accuracies = []
    f1_scores = []

    model.eval()
    device = resolve_device(device)
    model = model.to(device)

    if evaluator is None:
        evaluator = _yield_CD_evaluation(model, data_loader, device=device)

    progress = utils.ProgressTimer(verbose=verbose, prefix=prefix)
    progress.tic(total_items=len(data_loader.dataset))

    for output in evaluator:

        prec, rec, acc, f1 = output
        precisions.append(prec)
        recalls.append(rec)
        accuracies.append(acc)
        f1_scores.append(f1)

        progress.toc(add=1)

        if dry_run:
            break

    R = {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "accuracy": np.mean(accuracies),
        "f1_score": np.mean(f1_scores),
    }

    details = {
        "precision": precisions,
        "recall": recalls,
        "accuracy": accuracies,
        "f1_score": f1_scores,
    }

    output = (R,)

    if return_duration:
        output += (progress.total_seconds,)

    if return_details:
        output += (details,)

    return output
