# -*- coding: utf-8 -*-
"""
HDF5 data management and file utilities for quantum control experiments.

Provides SlabFile (enhanced HDF5 wrapper), AttrDict (attribute-access dict),
and file indexing helpers for sequential experiment data files.

Original code by Phil Reinhold & David Schuster, from the slab repository.
"""

import numpy as np
import h5py
import json
from pathlib import Path


# ── File indexing utilities ──────────────────────────────────────────────────


def next_file_index(datapath, prefix, suffix=""):
    """Find the next available file index for files matching prefix_*suffix."""
    dirlist = sorted(Path(datapath).glob(prefix + "_*" + suffix))
    if len(dirlist) > 0:
        try:
            ii = int(dirlist[-1].name.split("_")[-1][0:-3]) + 1
        except ValueError:
            ii = 0
    else:
        ii = 0
    return ii


def get_next_filename(datapath, prefix, suffix=""):
    """Generate the next sequential filename: prefix_00001.suffix"""
    ii = next_file_index(datapath, prefix, suffix)
    return prefix + "_%05d" % (ii) + suffix


# ── SlabFile: HDF5 wrapper for scientific data ───────────────────────────────


class SlabFile(h5py.File):
    """Enhanced HDF5 file wrapper for scientific data storage.

    Extends h5py.File with convenience methods for adding datasets,
    loading configs, and reading group attributes. Designed to be used
    like a Python dictionary where keys map to numpy arrays.

    Example::

        with SlabFile('test.h5', 'a') as f:
            f.add('xpts', np.linspace(0, 2*np.pi, 100))
            f.attrs['description'] = "Sine sweep data"
    """

    def __init__(self, *args, **kwargs):
        h5py.File.__init__(self, *args, **kwargs)
        self.flush()

    def add_data(self, f, key, data):
        """Add or replace a dataset in the given file/group.

        Creates a resizable dataset. If the key already exists, the old
        dataset is deleted and recreated.

        Args:
            f: HDF5 file or group to write into.
            key: Dataset name.
            data: Array-like data (converted to numpy array).
        """
        data = np.array(data)
        if key in f:
            del f[key]
        f.create_dataset(
            key,
            shape=data.shape,
            maxshape=tuple([None] * len(data.shape)),
            dtype=str(data.dtype),
        )
        f[key][...] = data

    def add(self, key, data):
        """Add or replace a dataset in this file (convenience wrapper)."""
        self.add_data(self, key, data)

    def get_dict(self, group="/"):
        """Return all attributes of a group as a plain dict.

        Args:
            group: HDF5 group path (default: root).
        """
        g = self[group]
        return {k: g.attrs[k] for k in g.attrs}

    # Alias for backward compatibility
    get_attrs = get_dict

    def load_config(self):
        """Load experiment config stored in the root 'config' attribute.

        Returns:
            AttrDict of the config, or None if no config is stored.
        """
        if "config" in self.attrs:
            return AttrDict(json.loads(self.attrs["config"]))
        return None


# ── AttrDict: dictionary with attribute-style access ─────────────────────────


class AttrDict(dict):
    """Dict subclass allowing attribute-style access (cfg.device.qubit.f_ge).

    Nested plain dicts are automatically wrapped in AttrDict on get/set
    so that attribute access works at every level.
    """

    def __init__(self, value=None):
        super().__init__()
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError("expected dict")

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        v = super().__getitem__(key)
        if isinstance(v, dict) and not isinstance(v, AttrDict):
            return AttrDict(v)
        return v

    def __setattr__(self, a, v):
        return self.__setitem__(a, v)

    def __getattr__(self, a):
        if a in self:
            return self.__getitem__(a)
        return self.__getattribute__(a)

    def to_dict(self):
        """Recursively convert back to plain nested dicts."""
        d = {}
        for k, v in self.items():
            d[k] = v.to_dict() if isinstance(v, AttrDict) else v
        return d
