# -*- coding: utf-8 -*-
"""
Data management utilities for quantum control experiments using HDF5.

This module provides enhanced interfaces to HDF5 file operations through 
the SlabFile class and related utilities for scientific data storage.

Original code by Phil Reinhold & David Schuster
Taken from the slab repository

:Authors: Phil Reinhold & David Schuster

The preferred format for saving data permanently is the
:py:class:`SlabFile`. This is a thin wrapper around the h5py_
interface to the HDF5_ file format. Using a SlabFile is much like
using a traditional python dictionary_, where the keys are strings,
and the values are `numpy arrays`_. A typical session using SlabFiles
in this way might look like this::

  import numpy as np
  from slab.datamanagement import SlabFile

  f = SlabFile('test.h5')
  f['xpts'] = np.linspace(0, 2*np.pi, 100)
  f['ypts'] = np.sin(f['xpts'])
  f.attrs['description'] = "One period of the sine function"

Notice several features of this interaction.

1. Numpy arrays are inserted directly into the file by assignment, no function calls needed
2. Datasets are retrieved from the file and used as you would a numpy array
3. Non-array elements can be saved in the file with the aid of the 'attrs' dictionary

.. _numpy arrays: http://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html
.. _dictionary: http://docs.python.org/2/tutorial/datastructures.html#dictionaries
.. _HDF5: http://www.hdfgroup.org/HDF5/
.. _h5py: https://code.google.com/p/h5py/
"""

import numpy as np
import h5py
import inspect
import datetime
import json
from pathlib import Path
import copy


class h5File(h5py.File):
    """
    Basic HDF5 file wrapper extending h5py.File with convenient data operations.
    
    This class provides simplified methods for adding and appending data to HDF5
    files with automatic dataset creation and resizing capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize h5File with the same arguments as h5py.File."""
        h5py.File.__init__(self, *args, **kwargs)

    def add(self, key, data):
        """
        Add or replace data in the file with automatic dataset creation.
        
        Creates a new dataset with unlimited maxshape for future expansion.
        If the dataset already exists, it's deleted and recreated.
        
        Args:
            key (str): Dataset name/key
            data (array-like): Data to store (will be converted to numpy array)
        """
        data = np.array(data)
        try:
            # Create dataset with unlimited dimensions for future resizing
            self.create_dataset(
                key,
                shape=data.shape,
                maxshape=tuple([None] * len(data.shape)),
                dtype=str(data.dtype),
            )
        except RuntimeError:
            # If dataset exists, delete and recreate
            del self[key]
            self.create_dataset(
                key,
                shape=data.shape,
                maxshape=tuple([None] * len(data.shape)),
                dtype=str(data.dtype),
            )
        # Write the data
        self[key][...] = data

    def append(self, key, data, forceInit=False):
        """
        Append data to an existing dataset or create new one if it doesn't exist.
        
        Creates datasets with an extra dimension for appending successive data.
        The first dimension grows with each append operation.
        
        Args:
            key (str): Dataset name/key
            data (array-like): Data to append (will be converted to numpy array)
            forceInit (bool): If True, reinitialize dataset even if it exists
        """
        data = np.array(data)
        try:
            # Create dataset with extra dimension for appending
            # Shape is (1, *data.shape) to allow for multiple appends
            self.create_dataset(
                key,
                shape=tuple([1] + list(data.shape)),
                maxshape=tuple([None] * (len(data.shape) + 1)),
                dtype=str(data.dtype),
            )
        except RuntimeError:
            if forceInit == True:
                # Force recreation of dataset
                del self[key]
                self.create_dataset(
                    key,
                    shape=tuple([1] + list(data.shape)),
                    maxshape=tuple([None] * (len(data.shape) + 1)),
                    dtype=str(data.dtype),
                )
            else:
                # Resize existing dataset to accommodate new data
                dataset = self[key]
                Shape = list(dataset.shape)
                Shape[0] = Shape[0] + 1  # Increase first dimension
                dataset.resize(Shape)

        # Write data to the last position
        dataset = self[key]
        try:
            dataset[-1, :] = data  # For multi-dimensional data
        except TypeError:
            dataset[-1] = data  # For 1D data
            # Note: All appended data must have same dimensionality


class SlabFile(h5py.File):
    """
    Enhanced HDF5 file wrapper with scientific data management features.
    
    SlabFile extends h5py.File with additional functionality commonly needed
    for scientific data storage including:
    - Dictionary-like attribute storage and retrieval
    - Convenient data appending operations
    - Timestamped notes and logging
    - Configuration management
    - Axis labeling and range setting for datasets
    
    Designed to be used like a Python dictionary where keys are dataset names
    and values are numpy arrays.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize SlabFile with immediate flush for data safety.
        
        Args:
            *args, **kwargs: Same arguments as h5py.File
        """
        h5py.File.__init__(self, *args, **kwargs)
        self.flush()  # Ensure immediate write to disk

    # Methods for proxy/remote use
    def _my_ds_from_path(self, dspath):
        """
        Navigate to dataset or group using hierarchical path.
        
        Args:
            dspath (list): List of strings representing path components
            
        Returns:
            HDF5 object (dataset or group) at the specified path
        """
        branch = self
        for ds in dspath:
            branch = branch[ds]
        return branch

    def _my_assign_dset(self, dspath, ds, val):
        """
        Assign value to dataset at specified path (for proxy use).
        
        Args:
            dspath (list): Path to parent group
            ds (str): Dataset name
            val: Value to assign
        """
        print("assigning", ds, val)
        branch = self._my_ds_from_path(dspath)
        branch[ds] = val

    def _get_dset_array(self, dspath):
        """
        Get pickle-safe array representation for remote access.
        
        Args:
            dspath (list): Path to dataset/group
            
        Returns:
            For groups: "group" string
            For datasets: Tuple of (H5Array, attributes dict)
        """
        branch = self._my_ds_from_path(dspath)
        if isinstance(branch, h5py.Group):
            return "group"
        else:
            return (H5Array(branch), dict(branch.attrs))

    def _get_attrs(self, dspath):
        """
        Get attributes dictionary for object at path.
        
        Args:
            dspath (list): Path to object
            
        Returns:
            dict: Attributes as key-value pairs
        """
        branch = self._my_ds_from_path(dspath)
        return dict(branch.attrs)

    def _set_attr(self, dspath, item, value):
        """
        Set attribute for object at path.
        
        Args:
            dspath (list): Path to object
            item (str): Attribute name
            value: Attribute value
        """
        branch = self._my_ds_from_path(dspath)
        branch.attrs[item] = value

    def _call_with_path(self, dspath, method, args, kwargs):
        """
        Call method on object at path (for proxy use).
        
        Args:
            dspath (list): Path to object
            method (str): Method name to call
            args (tuple): Method arguments
            kwargs (dict): Method keyword arguments
            
        Returns:
            Result of method call
        """
        branch = self._my_ds_from_path(dspath)
        return getattr(branch, method)(*args, **kwargs)

    def _ping(self):
        """
        Simple connectivity test for proxy connections.
        
        Returns:
            str: "OK" if connection is working
        """
        return "OK"

    def set_range(self, dataset, xmin, xmax, ymin=None, ymax=None):
        """
        Set axis range information for a dataset.
        
        Stores axis range metadata in dataset attributes for plotting/analysis.
        
        Args:
            dataset: HDF5 dataset object
            xmin, xmax (float): X-axis range
            ymin, ymax (float, optional): Y-axis range for 2D data
        """
        if ymin is not None and ymax is not None:
            dataset.attrs["_axes"] = ((xmin, xmax), (ymin, ymax))
        else:
            dataset.attrs["_axes"] = (xmin, xmax)

    def set_labels(self, dataset, x_lab, y_lab, z_lab=None):
        """
        Set axis labels for a dataset.
        
        Stores axis label metadata in dataset attributes for plotting/analysis.
        
        Args:
            dataset: HDF5 dataset object  
            x_lab (str): X-axis label
            y_lab (str): Y-axis label
            z_lab (str, optional): Z-axis label for 3D data
        """
        if z_lab is not None:
            dataset.attrs["_axes_labels"] = (x_lab, y_lab, z_lab)
        else:
            dataset.attrs["_axes_labels"] = (x_lab, y_lab)

    def append_line(self, dataset, line, axis=0):
        """
        Append a line of data to a 2D dataset.
        
        Creates dataset if it doesn't exist, then appends data along specified axis.
        Useful for building up 2D data arrays line by line during acquisition.
        
        Args:
            dataset (str or dataset): Dataset name or object
            line (array-like): 1D array of data to append
            axis (int): Axis along which to append (0 for rows, 1 for columns)
        """
        if isinstance(dataset, str):
            dataset = str(dataset)
        if isinstance(dataset, str):
            try:
                dataset = self[dataset]
            except:
                # Create new dataset with appropriate shape
                shape, maxshape = (0, len(line)), (None, len(line))
                if axis == 1:
                    shape, maxshape = (shape[1], shape[0]), (maxshape[1], maxshape[0])
                self.create_dataset(
                    dataset, shape=shape, maxshape=maxshape, dtype="float64"
                )
                dataset = self[dataset]
        
        # Resize dataset to accommodate new line
        shape = list(dataset.shape)
        shape[axis] = shape[axis] + 1
        dataset.resize(shape)
        
        # Add the new line
        if axis == 0:
            dataset[-1, :] = line  # Append as new row
        else:
            dataset[:, -1] = line  # Append as new column
        self.flush()

    def append_pt(self, dataset, pt):
        """
        Append a single data point to a 1D dataset.
        
        Creates dataset if it doesn't exist, then appends the point.
        Useful for building up time series or similar 1D data.
        
        Args:
            dataset (str or dataset): Dataset name or object
            pt (scalar): Data point to append
        """
        if isinstance(dataset, str):
            dataset = str(dataset)
        if isinstance(dataset, str):
            try:
                dataset = self[dataset]
            except:
                # Create new 1D dataset
                self.create_dataset(
                    dataset, shape=(0,), maxshape=(None,), dtype="float64"
                )
                dataset = self[dataset]
        
        # Resize and add point
        shape = list(dataset.shape)
        shape[0] = shape[0] + 1
        dataset.resize(shape)
        dataset[-1] = pt
        self.flush()

    def append_dset_pt(self, dataset, pt):
        """
        Append a point directly to a dataset object.
        
        Lower-level method for appending to an existing dataset object.
        
        Args:
            dataset: HDF5 dataset object
            pt (scalar): Data point to append
        """
        shape = dataset.shape[0]
        shape = shape + 1
        dataset.resize((shape,))
        dataset[-1] = pt
        dataset.flush()

    def note(self, note):
        """
        Add a timestamped note to HDF file in a dataset called 'notes'.
        
        Useful for logging experiment conditions, observations, or metadata
        that should be preserved with the data.
        
        Args:
            note (str): Note text to store
        """
        ts = datetime.datetime.now()
        try:
            ds = self["notes"]
        except:
            # Create notes dataset if it doesn't exist
            ds = self.create_dataset(
                "notes", (0,), maxshape=(None,), dtype=h5py.new_vlen(str)
            )

        # Add timestamped note
        shape = list(ds.shape)
        shape[0] = shape[0] + 1
        ds.resize(shape)
        ds[-1] = str(ts) + " -- " + note
        self.flush()

    def get_notes(self, one_string=False, print_notes=False):
        """
        Retrieve notes embedded in HDF file.
        
        Args:
            one_string (bool): If True, concatenate all notes into single string
            print_notes (bool): If True, print all notes to stdout
            
        Returns:
            list or str: Notes as list of strings or single concatenated string
        """
        try:
            notes = list(self["notes"])
        except:
            notes = []
            
        if print_notes:
            print("\n".join(notes))
            
        if one_string:
            notes = "\n".join(notes)
            
        return notes

    def add_data(self, f, key, data):
        """
        Add data to specified file with automatic dataset creation.
        
        Internal method used by add(). Creates resizable datasets.
        
        Args:
            f: HDF5 file object
            key (str): Dataset name
            data (array-like): Data to store
        """
        data = np.array(data)
        try:
            f.create_dataset(
                key,
                shape=data.shape,
                maxshape=tuple([None] * len(data.shape)),
                dtype=str(data.dtype),
            )
        except RuntimeError:
            # Dataset exists, delete and recreate
            del f[key]
            f.create_dataset(
                key,
                shape=data.shape,
                maxshape=tuple([None] * len(data.shape)),
                dtype=str(data.dtype),
            )
        f[key][...] = data

    def append_data(self, f, key, data, forceInit=False):
        """
        Append multi-dimensional data to a dataset in specified file.
        
        The main difference between append_pt and append is that
        append handles higher dimensional data, while append_pt handles only 1D.
        
        Args:
            f: HDF5 file object
            key (str): Dataset name
            data (array-like): Data to append
            forceInit (bool): If True, recreate dataset even if it exists
        """
        data = np.array(data)
        try:
            # Create with extra first dimension for appending
            f.create_dataset(
                key,
                shape=tuple([1] + list(data.shape)),
                maxshape=tuple([None] * (len(data.shape) + 1)),
                dtype=str(data.dtype),
            )
        except RuntimeError:
            if forceInit == True:
                # Force recreation of dataset
                del f[key]
                f.create_dataset(
                    key,
                    shape=tuple([1] + list(data.shape)),
                    maxshape=tuple([None] * (len(data.shape) + 1)),
                    dtype=str(data.dtype),
                )
            else:
                # Resize existing dataset
                dataset = f[key]
                Shape = list(dataset.shape)
                Shape[0] = Shape[0] + 1
                dataset.resize(Shape)

        # Write data to last position
        dataset = f[key]
        try:
            dataset[-1, :] = data  # Multi-dimensional data
        except TypeError:
            dataset[-1] = data  # 1D data
            # Note: All appended data must have same dimensionality

    def add(self, key, data):
        """Convenience wrapper for add_data() using self as file."""
        self.add_data(self, key, data)

    def append(self, dataset, pt):
        """Convenience wrapper for append_data() using self as file."""
        self.append_data(self, dataset, pt)

    def save_dict(self, dict, group="/"):
        """
        Save a dictionary to HDF5 group attributes.
        
        Args:
            dict (dict): Dictionary to save
            group (str): HDF5 group path (default: root)
        """
        if group not in self:
            self.create_group(group)
        for k in list(dict.keys()):
            self[group].attrs[k] = dict[k]

    def get_dict(self, group="/"):
        """
        Retrieve group attributes as a dictionary.
        
        Args:
            group (str): HDF5 group path (default: root)
            
        Returns:
            dict: Group attributes as key-value pairs
        """
        d = {}
        g = self[group]
        for k in g.attrs:
            d[k] = g.attrs[k]
        return d

    # Create aliases for common operations
    get_attrs = get_dict
    save_attrs = save_dict

    def get_group_data(self, group="/"):
        """
        Load all data and attributes from a group.
        
        Args:
            group (str): HDF5 group path (default: root)
            
        Returns:
            dict: Dictionary containing:
                  - 'attrs': group attributes
                  - dataset names as keys with numpy arrays as values
        """
        data = {"attrs": self.get_dict(group)}

        g = self[group]
        for k in g.keys():
            data[k] = np.array(g[k])
        return data

    def save_settings(self, dic, group="settings"):
        """
        Save settings dictionary to 'settings' group.
        
        Convenience method for experiment settings storage.
        
        Args:
            dic (dict): Settings dictionary to save
            group (str): Group name (default: "settings")
        """
        self.save_dict(dic, group)

    def load_settings(self, group="settings"):
        """
        Load settings from 'settings' group.
        
        Args:
            group (str): Group name (default: "settings")
            
        Returns:
            dict: Settings dictionary
        """
        return self.get_dict(group)

    def load_config(self):
        """
        Load experiment configuration from file attributes.
        
        Returns:
            AttrDict or None: Configuration as AttrDict if present, None otherwise
        """
        if "config" in list(self.attrs.keys()):
            return AttrDict(json.loads(self.attrs["config"]))
        else:
            return None


def set_range(dset, range_dsets, range_names=None):
    """
    usage:
        ds['x'] = linspace(0, 10, 100)
        ds['y'] = linspace(0, 1, 10)
        ds['z'] = [ sin(x*y) for x in ds['x'] for y in ds['y'] ]
        set_range(ds['z'], (ds['x'], ds['y']), ('x', 'y'))
    """
    for i, range_ds in enumerate(range_dsets):
        dset.dims.create_scale(range_ds)
        dset.dims[i].attach_scale(range_ds)
        if range_names:
            dset.dims[i].label = range_names[i]


def get_script():
    """returns currently running script file as a string"""
    fname = inspect.stack()[-1][1]
    if fname == "<stdin>":
        return fname
    # print fname
    f = open(fname, "r")
    s = f.read()
    f.close()
    return s


def open_to_path(h5file, path, pathsep="/"):
    f = h5file
    for name in path.split(pathsep):
        if name:
            f = f[name]
    return f


def get_next_trace_number(h5file, last=0, fmt="%03d"):
    i = last
    while (fmt % i) in h5file:
        i += 1
    return i


def open_to_next_trace(h5file, last=0, fmt="%03d"):
    return h5file[fmt % get_next_trace_number(h5file, last, fmt)]


def load_array(f, array_name):
    if f[array_name].len() == 0:
        a = []
    else:
        a = np.zeros(f[array_name].shape)
        f[array_name].read_direct(a)

    return a


def load_slabfile_data(fname, path="", group="/"):
    fullname = str(Path(path) / fname)
    with SlabFile(fullname, "r") as f:
        data = f.get_group_data(group)
    return data


class AttrDict(dict):
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
        super(AttrDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        v = super().__getitem__(key)
        if isinstance(v, dict) and not isinstance(v, AttrDict):
            return AttrDict(v)
        else:
            return v

    def __setattr__(self, a, v):
        return self.__setitem__(a, v)

    def __getattr__(self, a):
        if a in self:
            return self.__getitem__(a)
        else:
            return self.__getattribute__(a)  # @IgnoreException

    def to_dict(self):
        d = {}
        for k, v in self.items():
            if isinstance(v, AttrDict):
                d[k] = v.to_dict()
            else:
                d[k] = v
        return d

    # def __deepcopy__(self, memo):
    #     # Deepcopy only the id attribute, then construct the new instance and map
    #     # the id() of the existing copy to the new instance in the memo dictionary
    #     memo[id(self)] = newself = self.__class__(copy.deepcopy(self.id, memo))
    #     # Now that memo is populated with a hashable instance, copy the other attributes:
    #     newself.degree = copy.deepcopy(self.degree, memo)
    #     # Safe to deepcopy edge_dict now, because backreferences to self will
    #     # be remapped to newself automatically
    #     newself.edge_dict = copy.deepcopy(self.edge_dict, memo)
    #     return newself

    # def __new__(cls, p_id):
    #     self = super().__new__(cls)  # Must explicitly create the new object
    #     # Aside from explicit construction and return, rest of __new__
    #     # is same as __init__
    #     self.id = p_id
    #     self.edge_dict = {}
    #     self.degree = 0
    #     return self  # __new__ returns the new object

    # def __getnewargs__(self):
    #     # Return the arguments that *must* be passed to __new__
    #     return (self.id,)
