""" NYUv2 dataset object. """

import os
import h5py
import math
import shutil
import random
import tarfile
import zipfile
from PIL import Image
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets.utils import download_url

from meta.datasets import BaseDataset
from meta.datasets.utils import RGB_TRANSFORM, get_split
from meta.train.loss import CosineSimilarityLoss, MultiTaskLoss, ScaleInvariantDepthLoss


# This is HxW, as is expected by transforms.Resize.
IMG_SIZE = (3, 480, 640)
SEED = 1
EPSILON = 1e-5

DEPTH_TRANSFORM = transforms.ToTensor()
SEG_TRANSFORM = transforms.ToTensor()
SN_TRANSFORM = transforms.ToTensor()

NUM_TASKS = 3


class NYUv2(Dataset, BaseDataset):
    """
    Wrapper for the NYUv2 dataset. Data sources available: RGB, Semantic Segmentation,
    Surface Normals, Depth Images.
    """

    def __init__(
        self,
        root: str,
        task_type: str,
        train: bool = True,
        download: bool = False,
        min_crop_ratio: float = 1.0,
        jitter_factor: float = 0.0,
        scale: float = 1,
    ):
        """
        Will return tuples based on what data source has been enabled (rgb, seg etc).

        :param root: path to root folder (eg /data/NYUv2)
        :param task_type: what type of task, one of: ["seg", "sn", "depth", "multitask"]
        :param train: whether to load the train or test set
        :param download: whether to download and process data if missing
        :param min_crop_ratio: minimum of ratio of size of randomly cropped image to
        original image.  Should be between 0 and 1. If 1, no cropping is performed.
        :param jitter_factor: Factor by which to jitter brightness, constrast,
        saturation, and hue. Should be between 0 and 1. If 0,
        :param scale: how to scale images/labels. If 1 image sizes will not be changed,
        if 0.5 each dimension will be reduced by a factor of 2, etc.
        """

        assert task_type in ["seg", "sn", "depth", "multitask"]
        assert 0.0 <= min_crop_ratio <= 1.0
        assert 0.0 <= jitter_factor <= 1.0

        Dataset.__init__(self)
        BaseDataset.__init__(self)

        # Store data settings.
        self.root = root
        self.train = train
        self._split = "train" if train else "test"
        self.task_type = task_type
        self.scale = scale
        self.spatial_size = (
            round(IMG_SIZE[1] * self.scale),
            round(IMG_SIZE[2] * self.scale),
        )
        self.min_crop_ratio = min_crop_ratio
        self.random_crop = self.min_crop_ratio != 1.0
        self.jitter_factor = jitter_factor
        self.jitter = self.jitter_factor != 0.0

        # Store which tasks are being used.
        self.rgb = True
        self.seg = self.task_type in ["seg", "multitask"]
        self.sn = self.task_type in ["sn", "multitask"]
        self.depth = self.task_type in ["depth", "multitask"]

        # Construct transforms for each data type.
        self.rgb_transform = RGB_TRANSFORM
        self.seg_transform = None
        self.sn_transform = None
        self.depth_transform = None

        if self.scale != 1.0:
            self.rgb_scale = transforms.Resize(self.spatial_size)

        if self.task_type in ["seg", "multitask"]:
            self.seg_transform = SEG_TRANSFORM
            if self.scale != 1.0:
                self.seg_scale = transforms.Resize(
                    self.spatial_size, interpolation=InterpolationMode.NEAREST
                )

        if self.task_type in ["sn", "multitask"]:
            self.sn_transform = SN_TRANSFORM
            if self.scale != 1.0:
                self.sn_scale = transforms.Resize(self.spatial_size)

        if self.task_type in ["depth", "multitask"]:
            self.depth_transform = DEPTH_TRANSFORM
            if self.scale != 1.0:
                self.depth_scale = transforms.Resize(self.spatial_size)

        # Add jitter to transforms, if necessary.
        if self.jitter:
            self.rgb_transform = transforms.Compose(
                [
                    transforms.ColorJitter(
                        brightness=self.jitter_factor,
                        contrast=self.jitter_factor,
                        saturation=self.jitter_factor,
                        hue=self.jitter_factor / 2.0,
                    ),
                    self.rgb_transform,
                ]
            )

        # Set static dataset properties.
        if self.task_type == "seg":
            out_channels = 13
            self.loss_cls = nn.CrossEntropyLoss
            self.loss_kwargs = {"ignore_index": -1, "reduction": "mean"}
            self.extra_metrics = {
                "pixel_accuracy": {
                    "maximize": True,
                    "train": True,
                    "eval": True,
                    "show": True,
                },
            }
        elif self.task_type == "sn":
            out_channels = 3
            self.loss_cls = CosineSimilarityLoss
            self.loss_kwargs = {"reduction": "mean"}
            self.extra_metrics = {
                "accuracy@11.25": {
                    "maximize": True,
                    "train": True,
                    "eval": True,
                    "show": True,
                },
                "angle": {
                    "maximize": False,
                    "train": True,
                    "eval": True,
                    "show": False,
                },
            }
        elif self.task_type == "depth":
            out_channels = 1
            self.loss_cls = ScaleInvariantDepthLoss
            self.loss_kwargs = {"alpha": 0.5, "reduction": "mean"}
            self.extra_metrics = {
                "accuracy@1.25": {
                    "maximize": True,
                    "train": True,
                    "eval": True,
                    "show": True,
                },
                "RMSE": {"maximize": False, "train": True, "eval": True, "show": False},
            }
        elif self.task_type == "multitask":
            out_channels = 17
            self.loss_cls = MultiTaskLoss
            self.loss_kwargs = {
                "task_losses": [
                    {
                        "loss": nn.CrossEntropyLoss(ignore_index=-1, reduction="mean"),
                        "output_slice": get_seg_outputs,
                        "label_slice": get_seg_labels,
                    },
                    {
                        "loss": CosineSimilarityLoss(reduction="mean"),
                        "output_slice": get_sn_outputs,
                        "label_slice": get_sn_labels,
                    },
                    {
                        "loss": ScaleInvariantDepthLoss(alpha=0.5, reduction="mean"),
                        "output_slice": get_depth_outputs,
                        "label_slice": get_depth_labels,
                    },
                ]
            }
            self.criterion_kwargs = {"train": {"train": True}, "eval": {"train": False}}
            self.extra_metrics = {
                "seg_pixel_accuracy": {
                    "maximize": True,
                    "train": True,
                    "eval": True,
                    "show": False,
                },
                "sn_accuracy@11.25": {
                    "maximize": True,
                    "train": True,
                    "eval": True,
                    "show": False,
                },
                "sn_angle": {
                    "maximize": False,
                    "train": True,
                    "eval": True,
                    "show": False,
                },
                "depth_accuracy@1.25": {
                    "maximize": True,
                    "train": True,
                    "eval": True,
                    "show": False,
                },
                "depth_RMSE": {
                    "maximize": False,
                    "train": True,
                    "eval": True,
                    "show": False,
                },
                "avg_accuracy": {
                    "maximize": True,
                    "train": True,
                    "eval": True,
                    "show": True,
                },
                "var_accuracy": {
                    "maximize": None,
                    "train": True,
                    "eval": True,
                    "show": False,
                },
                **{
                    f"loss_weight_{t}": {
                        "maximize": None,
                        "train": True,
                        "eval": False,
                        "show": False,
                    }
                    for t in range(NUM_TASKS)
                },
            }
        else:
            assert False

        self.input_size = (
            IMG_SIZE[0],
            round(IMG_SIZE[1] * self.scale),
            round(IMG_SIZE[2] * self.scale),
        )
        self.output_size = (out_channels, self.input_size[1], self.input_size[2])

        # Download dataset if necessary, and check whether it exists.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not complete." + " You can use download=True to download it"
            )

        # rgb folder as ground truth
        self._files = os.listdir(os.path.join(root, f"{self._split}_rgb"))

        random.seed(SEED)

    def __getitem__(self, index: int):
        folder = lambda name: os.path.join(self.root, f"{self._split}_{name}")
        input_img = None
        labels = []

        if self.random_crop:
            crop_ratio = self.min_crop_ratio + random.random() * (
                1 - self.min_crop_ratio
            )
            crop_height = round(IMG_SIZE[1] * crop_ratio)
            crop_width = round(IMG_SIZE[2] * crop_ratio)
            crop_top = random.randrange(IMG_SIZE[1] - crop_height + 1)
            crop_left = random.randrange(IMG_SIZE[2] - crop_width + 1)
            crop_bottom = crop_top + crop_height
            crop_right = crop_left + crop_width

        if self.rgb:
            img = Image.open(os.path.join(folder("rgb"), self._files[index]))
            img = self.rgb_transform(img)
            if self.random_crop:
                img = img[:, crop_top:crop_bottom, crop_left:crop_right]
            if self.scale != 1.0:
                img = self.rgb_scale(img)

            input_img = img

        if self.seg:
            img = Image.open(os.path.join(folder("seg13"), self._files[index]))
            img = self.seg_transform(img)
            if self.random_crop:
                img = img[:, crop_top:crop_bottom, crop_left:crop_right]
            if self.scale != 1.0:
                img = self.seg_scale(img)
            if isinstance(img, torch.Tensor):
                # ToTensor scales to [0, 1] by default
                img = (img * 255).long()

            # Class 0 counts as an unlabeled pixel. We use -1 as the unlabeled value instead.
            img -= 1

            labels.append(img)

        if self.sn:
            img = Image.open(os.path.join(folder("sn"), self._files[index]))
            img = self.sn_transform(img)
            if self.random_crop:
                img = img[:, crop_top:crop_bottom, crop_left:crop_right]
            if self.scale != 1.0:
                img = self.sn_scale(img)
            labels.append(img)

        if self.depth:
            img = Image.open(os.path.join(folder("depth"), self._files[index]))
            img = np.array(img, dtype=np.float32) / 1e4
            img = self.depth_transform(img)
            if self.random_crop:
                img = img[:, crop_top:crop_bottom, crop_left:crop_right]
            if self.scale != 1.0:
                img = self.depth_scale(img)
            img = torch.maximum(img, EPSILON * torch.ones_like(img))
            labels.append(img)

        # Reduce labels list if there is only a single label, otherwise combine into a
        # single tensor.
        if len(labels) == 1:
            labels = labels[0]
        else:
            labels = torch.cat(labels, dim=0)

        # Unsqueeze class dimension of semantic segmentation label if it is the only one
        # being loaded.
        if self.task_type == "seg":
            labels = labels.squeeze(0)

        return input_img, labels

    def __len__(self):
        return len(self._files)

    def __repr__(self):
        fmt_str = f"Dataset {self.__class__.__name__}\n"
        fmt_str += f"    Number of data points: {self.__len__()}\n"
        fmt_str += f"    Split: {self._split}\n"
        fmt_str += f"    Root Location: {self.root}\n"
        tmp = "    RGB Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.rgb_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Seg Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.seg_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    SN Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.sn_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Depth Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.depth_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

    def _check_exists(self) -> bool:
        """
        Only checking for folder existence
        """
        try:
            for split in ["train", "test"]:
                for type_ in ["rgb", "seg13", "sn", "depth"]:
                    path = os.path.join(self.root, f"{split}_{type_}")
                    if not os.path.exists(path):
                        raise FileNotFoundError("Missing Folder")
        except FileNotFoundError:
            return False
        return True

    def download(self):
        if self._check_exists():
            return
        download_rgb(self.root)
        download_seg(self.root)
        download_sn(self.root)
        download_depth(self.root)
        print("Done!")

    def compute_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module = None,
        train: bool = True,
    ) -> Dict[str, float]:
        """ Compute metrics for any of the present tasks from `outputs` and `labels`. """

        split = get_split(train)
        if self.task_type == "seg":
            return {
                f"{split}_pixel_accuracy": NYUv2_seg_pixel_accuracy(outputs, labels)
            }

        elif self.task_type == "sn":
            return {
                f"{split}_accuracy@11.25": NYUv2_sn_accuracy(
                    outputs, labels, thresh=11.25
                ),
                f"{split}_angle": NYUv2_sn_angle(outputs, labels),
            }

        elif self.task_type == "depth":
            return {
                f"{split}_accuracy@1.25": NYUv2_depth_accuracy(
                    outputs, labels, thresh=1.25
                ),
                f"{split}_RMSE": NYUv2_depth_RMSE(outputs, labels),
            }

        elif self.task_type == "multitask":

            # Get outputs and labels from each task.
            seg_outputs = get_seg_outputs(outputs)
            seg_labels = get_seg_labels(labels)
            sn_outputs = get_sn_outputs(outputs)
            sn_labels = get_sn_labels(labels)
            depth_outputs = get_depth_outputs(outputs)
            depth_labels = get_depth_labels(labels)

            # Compute accuracy for each task, then mean and variances of accuracies.
            seg_acc = NYUv2_seg_pixel_accuracy(seg_outputs, seg_labels)
            sn_acc = NYUv2_sn_accuracy(sn_outputs, sn_labels, thresh=11.25)
            depth_acc = NYUv2_depth_accuracy(depth_outputs, depth_labels, thresh=1.25)
            task_accs = [seg_acc, sn_acc, depth_acc]
            avg_acc = np.mean(task_accs)
            var_acc = np.var(task_accs)

            metrics = {
                f"{split}_seg_pixel_accuracy": seg_acc,
                f"{split}_sn_accuracy@11.25": sn_acc,
                f"{split}_sn_angle": NYUv2_sn_angle(sn_outputs, sn_labels),
                f"{split}_depth_accuracy@1.25": depth_acc,
                f"{split}_depth_RMSE": NYUv2_depth_RMSE(depth_outputs, depth_labels),
                f"{split}_avg_accuracy": avg_acc,
                f"{split}_var_accuracy": var_acc,
            }

            # Add loss weights to metrics, if this is a train step.
            if train:
                assert isinstance(criterion, MultiTaskLoss)
                weight_tensor = criterion.loss_weighter.loss_weights
                loss_weights = {
                    f"{split}_loss_weight_{t}": float(weight_tensor[t])
                    for t in range(NUM_TASKS)
                }
                metrics.update(loss_weights)

            return metrics

        else:
            assert False


def download_rgb(root: str):
    train_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz"
    test_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz"

    def _proc(url: str, dst: str):
        if not os.path.exists(dst):
            tar = os.path.join(root, url.split("/")[-1])
            if not os.path.exists(tar):
                download_url(url, root)
            if os.path.exists(tar):
                _unpack(tar)
                _replace_folder(tar.rstrip(".tgz"), dst)
                _rename_files(dst, lambda x: x.split("_")[2])

    _proc(train_url, os.path.join(root, "train_rgb"))
    _proc(test_url, os.path.join(root, "test_rgb"))


def download_seg(root: str):
    train_url = "https://github.com/ankurhanda/nyuv2-meta-data/raw/master/train_labels_13/nyuv2_train_class13.tgz"
    test_url = "https://github.com/ankurhanda/nyuv2-meta-data/raw/master/test_labels_13/nyuv2_test_class13.tgz"

    def _proc(url: str, dst: str):
        if not os.path.exists(dst):
            tar = os.path.join(root, url.split("/")[-1])
            if not os.path.exists(tar):
                download_url(url, root)
            if os.path.exists(tar):
                _unpack(tar)
                _replace_folder(tar.rstrip(".tgz"), dst)
                _rename_files(dst, lambda x: x.split("_")[3])

    _proc(train_url, os.path.join(root, "train_seg13"))
    _proc(test_url, os.path.join(root, "test_seg13"))


def download_sn(root: str):
    url = "https://www.inf.ethz.ch/personal/ladickyl/nyu_normals_gt.zip"
    train_dst = os.path.join(root, "train_sn")
    test_dst = os.path.join(root, "test_sn")

    if not os.path.exists(train_dst) or not os.path.exists(test_dst):
        tar = os.path.join(root, url.split("/")[-1])
        if not os.path.exists(tar):
            download_url(url, root)
        if os.path.exists(tar):
            _unpack(tar)
            if not os.path.exists(train_dst):
                _replace_folder(
                    os.path.join(root, "nyu_normals_gt", "train"), train_dst
                )
                _rename_files(train_dst, lambda x: x[1:])
            if not os.path.exists(test_dst):
                _replace_folder(os.path.join(root, "nyu_normals_gt", "test"), test_dst)
                _rename_files(test_dst, lambda x: x[1:])
            shutil.rmtree(os.path.join(root, "nyu_normals_gt"))


def download_depth(root: str):
    url = (
        "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    )
    train_dst = os.path.join(root, "train_depth")
    test_dst = os.path.join(root, "test_depth")

    if not os.path.exists(train_dst) or not os.path.exists(test_dst):
        tar = os.path.join(root, url.split("/")[-1])
        if not os.path.exists(tar):
            download_url(url, root)
        if os.path.exists(tar):
            train_ids = [
                f.split(".")[0] for f in os.listdir(os.path.join(root, "train_rgb"))
            ]
            _create_depth_files(tar, root, train_ids)


def _unpack(file: str):
    """
    Unpacks tar and zip, does nothing for any other type
    :param file: path of file
    """
    path = file.rsplit(".", 1)[0]

    if file.endswith(".tgz"):
        tar = tarfile.open(file, "r:gz")
        tar.extractall(path)
        tar.close()
    elif file.endswith(".zip"):
        zip = zipfile.ZipFile(file, "r")
        zip.extractall(path)
        zip.close()


def _rename_files(folder: str, rename_func: callable):
    """
    Renames all files inside a folder based on the passed rename function
    :param folder: path to folder that contains files
    :param rename_func: function renaming filename (not including path) str -> str
    """
    imgs_old = os.listdir(folder)
    imgs_new = [rename_func(file) for file in imgs_old]
    for img_old, img_new in zip(imgs_old, imgs_new):
        shutil.move(os.path.join(folder, img_old), os.path.join(folder, img_new))


def _replace_folder(src: str, dst: str):
    """
    Rename src into dst, replacing/overwriting dst if it exists.
    """
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)


def _create_depth_files(mat_file: str, root: str, train_ids: list):
    """
    Extract the depth arrays from the mat file into images
    :param mat_file: path to the official labelled dataset .mat file
    :param root: The root directory of the dataset
    :param train_ids: the IDs of the training images as string (for splitting)
    """
    os.mkdir(os.path.join(root, "train_depth"))
    os.mkdir(os.path.join(root, "test_depth"))
    train_ids = set(train_ids)

    depths = h5py.File(mat_file, "r")["depths"]
    for i in range(len(depths)):
        img = (depths[i] * 1e4).astype(np.uint16).T
        id_ = str(i + 1).zfill(4)
        folder = "train" if id_ in train_ids else "test"
        save_path = os.path.join(root, f"{folder}_depth", id_ + ".png")
        Image.fromarray(img).save(save_path)


get_seg_outputs = lambda x: x[:, :13]
get_seg_labels = lambda x: x[:, 0].long()
get_sn_outputs = lambda x: x[:, 13:16]
get_sn_labels = lambda x: x[:, 1:4]
get_depth_outputs = lambda x: x[:, 16:17]
get_depth_labels = lambda x: x[:, 4:5]


def NYUv2_seg_pixel_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute pixel accuracy of semantic segmentation on the NYUv2 dataset. Here we assume
    that any pixels with label -1 are unlabeled, so we don't count these pixels in the
    accuracy computation. We also assume that the class dimension is directly after the
    batch dimension.
    """
    preds = torch.argmax(outputs, dim=1)
    correct = torch.sum(preds == labels)
    valid = torch.sum(labels != -1)
    accuracy = correct / valid
    return accuracy.item()


def NYUv2_seg_class_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute class accuracy of semantic segmentation on the NYUv2 dataset. Here we assume
    that any pixels with label -1 are unlabeled, so we don't count these pixels in the
    accuracy computation. We also assume that the class dimension is directly after the
    batch dimension.
    """

    # Get predictions.
    preds = torch.argmax(outputs, dim=1)

    # Get list of all labels in image.
    unlabel = -1
    all_labels = labels.unique().tolist()
    if unlabel in all_labels:
        all_labels.remove(unlabel)

    # Compute accuracy per-class.
    class_accuracies = torch.zeros(len(all_labels), device=outputs.device)
    for i, label in enumerate(all_labels):
        class_correct = torch.sum(torch.logical_and(preds == label, labels == label))
        class_valid = torch.sum(labels == label)
        class_accuracies[i] = class_correct / class_valid

    # Return average class accuracy.
    return class_accuracies.mean().item()


def NYUv2_seg_class_IOU(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute mean of IOU for each class of semantic segmentation on the NYUv2 dataset.
    Here we assume that any pixels with label -1 are unlabeled, so we don't count these
    pixels in the accuracy computation. We also assume that the class dimension is
    directly after the batch dimension.
    """

    # Get predictions.
    preds = torch.argmax(outputs, dim=1)

    # Get list of all labels in image.
    unlabel = -1
    all_labels = labels.unique().tolist()
    if unlabel in all_labels:
        all_labels.remove(unlabel)

    # Compute IOU per-class.
    class_IOUs = torch.zeros(len(all_labels), device=outputs.device)
    for i, label in enumerate(all_labels):
        class_intersection = torch.sum(
            torch.logical_and(preds == label, labels == label)
        )
        class_union = torch.sum(torch.logical_or(preds == label, labels == label))
        class_IOUs[i] = class_intersection / class_union

    # Return average class accuracy.
    return class_IOUs.mean().item()


def NYUv2_sn_accuracy(
    outputs: torch.Tensor, labels: torch.Tensor, thresh=11.25
) -> float:
    """
    Compute accuracy of surface normal estimation on the NYUv2 dataset. We define this
    as the number of pixels for which the angle between the true normal and the
    predicted normal is less than `thresh` degrees. Here we assume that the normal
    dimension is 1.
    """
    similarity_threshold = math.cos(thresh / 180 * math.pi)
    similarity = F.cosine_similarity(outputs, labels, dim=1)
    accuracy = torch.sum(similarity > similarity_threshold) / torch.numel(similarity)
    return accuracy.item()


def NYUv2_sn_angle(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute the mean angle between ground truth and predicted normal for the NYUv2
    dataset.
    """
    cos = F.cosine_similarity(outputs, labels, dim=1)
    cos = torch.clamp(cos, -1, 1)
    angle = torch.acos(cos) * 180 / math.pi
    return torch.mean(angle).item()


def NYUv2_depth_accuracy(
    outputs: torch.Tensor, labels: torch.Tensor, thresh=1.25
) -> float:
    """
    Compute accuracy of depth prediction on the NYUv2 dataset. We define this as the
    number of pixels for which the ratio between the predicted depth and the true depth
    is less than `thresh`. Note: `thresh` should be greater than 1.
    """
    preds = torch.exp(outputs)
    ratio = torch.max(preds / labels, labels / preds)
    accuracy = torch.sum(ratio < thresh) / torch.numel(ratio)
    return accuracy.item()


def NYUv2_depth_RMSE(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """ Root mean-square error for NYUv2 depth prediction. """
    preds = torch.exp(outputs)
    return torch.sqrt(torch.mean((preds - labels) ** 2)).item()


def NYUv2_depth_log_RMSE(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """ RMSE of log-prediction and log-ground truth for NYUv2 depth prediction. """
    return torch.sqrt(torch.mean((outputs - torch.log(labels)) ** 2)).item()


def NYUv2_depth_invariant_RMSE(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Scale-invariant RMSE of log-prediction and log-ground truth for NYUv2 depth
    prediction. Note that the network is trained to compute the log-depth (in
    `ScaleInvariantDepthLoss`).
    """
    diffs = outputs - torch.log(labels)
    mse = torch.mean(diffs ** 2)
    relative = torch.sum(diffs) ** 2 / torch.numel(diffs) ** 2
    return torch.sqrt(mse - relative).item()
