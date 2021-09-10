""" NYUv2 dataset object. """

import os
import sys
import h5py
import torch
import shutil
import random
import tarfile
import zipfile
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets.utils import download_url


# This is HxW, as is expected by transforms.Resize.
IMG_SIZE = (480, 640)
SEED = 1
EPSILON = 1e-5


class NYUv2(Dataset):
    """
    PyTorch wrapper for the NYUv2 dataset focused on multi-task learning. Data sources
    available: RGB, Semantic Segmentation, Surface Normals, Depth Images. If no
    transformation is provided, the image type will not be returned.

    ### Output
    All images are of size: 640 x 480 by default, though the dimensions can be scaled.

    1. RGB: 3 channel input image

    2. Semantic Segmentation: 1 channel representing one of the 14 (0 -
    background) classes. Conversion to int will happen automatically if
    transformation ends in a tensor.

    3. Surface Normals: 3 channels, with values in [0, 1].

    4. Depth Images: 1 channel with floats representing the distance in meters.
    Conversion will happen automatically if transformation ends in a tensor.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        rgb_transform=None,
        seg_transform=None,
        sn_transform=None,
        depth_transform=None,
        min_crop_ratio: float = 1.0,
        jitter_factor: float = 0.0,
        scale: float = 1,
    ):
        """
        Will return tuples based on what data source has been enabled (rgb, seg etc).

        :param root: path to root folder (eg /data/NYUv2)
        :param train: whether to load the train or test set
        :param download: whether to download and process data if missing
        :param rgb_transform: the transformation pipeline for rbg images
        :param seg_transform: the transformation pipeline for segmentation images. If
        the transformation ends in a tensor, the result will be automatically
        converted to int in [0, 14)
        :param sn_transform: the transformation pipeline for surface normal images
        :param depth_transform: the transformation pipeline for depth images. If the
        transformation ends in a tensor, the result will be automatically converted
        to meters
        :param min_crop_ratio: minimum of ratio of size of randomly cropped image to
        original image.  Should be between 0 and 1. If 1, no cropping is performed.
        :param jitter_factor: Factor by which to jitter brightness, constrast,
        saturation, and hue. Should be between 0 and 1. If 0,
        :param scale: how to scale images/labels. If 1 image sizes will not be changed,
        if 0.5 each dimension will be reduced by a factor of 2, etc.
        """
        super().__init__()
        self.root = root

        assert 0.0 <= min_crop_ratio <= 1.0
        assert 0.0 <= jitter_factor <= 1.0

        self.scale = scale
        self.size = (round(IMG_SIZE[0] * self.scale), round(IMG_SIZE[1] * self.scale))
        self.rgb_transform = rgb_transform
        self.seg_transform = seg_transform
        self.sn_transform = sn_transform
        self.depth_transform = depth_transform
        self.min_crop_ratio = min_crop_ratio
        self.random_crop = self.min_crop_ratio != 1.0
        self.jitter_factor = jitter_factor
        self.jitter = self.jitter_factor != 0.0

        # Construct scaling transforms if necessary.
        if self.rgb_transform is not None and self.scale != 1:
            self.rgb_scale = transforms.Resize(self.size)
        if self.seg_transform is not None and self.scale != 1:
            self.seg_scale = transforms.Resize(self.size, interpolation=InterpolationMode.NEAREST)
        if self.sn_transform is not None and self.scale != 1:
            self.sn_scale = transforms.Resize(self.size)
        if self.depth_transform is not None and self.scale != 1:
            self.depth_scale = transforms.Resize(self.size)

        # Add jitter to transforms, if necessary.
        if self.jitter:
            if self.rgb_transform is not None:
                self.rgb_transform = transforms.Compose([
                    transforms.ColorJitter(
                        brightness=self.jitter_factor,
                        contrast=self.jitter_factor,
                        saturation=self.jitter_factor,
                        hue=self.jitter_factor / 2.0,
                    ),
                    self.rgb_transform,
                ])

        self.train = train
        self._split = "train" if train else "test"

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

        rgb = self.rgb_transform is not None
        seg = self.seg_transform is not None
        sn = self.sn_transform is not None
        depth = self.depth_transform is not None

        if self.random_crop:
            crop_ratio = self.min_crop_ratio + random.random() * (1 - self.min_crop_ratio)
            crop_height = round(IMG_SIZE[0] * crop_ratio)
            crop_width = round(IMG_SIZE[1] * crop_ratio)
            crop_top = random.randrange(IMG_SIZE[0] - crop_height + 1)
            crop_left = random.randrange(IMG_SIZE[1] - crop_width + 1)
            crop_bottom = crop_top + crop_height
            crop_right = crop_left + crop_width

        if rgb:
            img = Image.open(os.path.join(folder("rgb"), self._files[index]))
            img = self.rgb_transform(img)
            if self.random_crop:
                img = img[:, crop_top:crop_bottom, crop_left:crop_right]
            if self.scale != 1.0:
                img = self.rgb_scale(img)

            input_img = img

        if seg:
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

        if sn:
            img = Image.open(os.path.join(folder("sn"), self._files[index]))
            img = self.sn_transform(img)
            if self.random_crop:
                img = img[:, crop_top:crop_bottom, crop_left:crop_right]
            if self.scale != 1.0:
                img = self.sn_scale(img)
            labels.append(img)

        if depth:
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
        if seg and not sn and not depth:
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
        except FileNotFoundError as e:
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
