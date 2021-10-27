""" Shared utilities for Trainer class and subclasses. """

from typing import Iterable, Any


def cycle(iterable: Iterable[Any]) -> Any:
    """
    Generator to repeatedly cycle through an iterable. This is a hacky way to get our
    batch sampling to work with the way our Trainer class is set up. In particular, the
    data loaders are stored as members of Trainer subclass and each call to `_step()`
    requires one sample from these data loaders. This means we can't just loop over the
    data loaders, we have to sample the next batch one at a time.
    """
    while True:
        for x in iterable:
            yield x
