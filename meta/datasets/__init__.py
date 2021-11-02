""" Dataset objects. """

from meta.datasets.base import BaseDataset
from meta.datasets.mnist import MNIST
from meta.datasets.cifar import CIFAR10, CIFAR100
from meta.datasets.nyuv2 import NYUv2
from meta.datasets.mtregression import MTRegression
from meta.datasets.pcba import PCBA
from meta.datasets.continual import ContinualDataset
from meta.datasets.rotated import Rotated

DATASETS = [
    "MNIST", "CIFAR10", "CIFAR100", "NYUv2", "MTRegression", "PCBA", "Rotated"
]
