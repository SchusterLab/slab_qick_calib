"""
Experiment handling framework for quantum control experiments.

This module provides the base Experiment class for managing quantum control experiments,
including data acquisition, analysis, storage, and configuration management.

Original code by David Schuster
Taken from the slab repository
"""

__author__ = "David Schuster"

import os.path
import json
import yaml
import numpy as np
import traceback

from .datamanagement import SlabFile, AttrDict
from .instrumentmanager import InstrumentManager
from .dataanalysis import get_next_filename


class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy data types and arrays.
    
    This encoder ensures that numpy arrays and scalar types can be properly
    serialized to JSON format for configuration and data storage.
    """

    def default(self, obj):
        """
        Convert numpy objects to JSON-serializable types.
        
        Args:
            obj: Object to be serialized
            
        Returns:
            JSON-serializable representation of the object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if str(type(obj)) == "<class 'qick.asm_v2.QickParam'>":
            return ""
        return super(NpEncoder, self).default(obj)


class YamlNpEncoder:
    """
    Custom YAML encoder to handle NumPy data types and arrays.
    
    This encoder ensures that numpy arrays and scalar types can be properly
    serialized to YAML format for configuration and data storage.
    Unlike JSON, YAML has native support for many data types, but we still
    need to convert numpy objects to standard Python types.
    """

    @staticmethod
    def convert_numpy_types(obj):
        """
        Recursively convert numpy objects to YAML-serializable types.
        
        Args:
            obj: Object to be converted (can be nested dict/list structure)
            
        Returns:
            YAML-serializable representation of the object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle special float values
            if np.isnan(obj):
                return None  # YAML represents NaN as null
            elif np.isinf(obj):
                return float('inf') if obj > 0 else float('-inf')
            else:
                return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: YamlNpEncoder.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [YamlNpEncoder.convert_numpy_types(item) for item in obj]
        elif str(type(obj)) == "<class 'qick.asm_v2.QickParam'>":
            return ""
        else:
            return obj

    @staticmethod
    def dump(data, stream=None, **kwargs):
        """
        Serialize data to YAML format with numpy type conversion.
        
        Args:
            data: Data to serialize
            stream: Output stream (file-like object) or None for string output
            **kwargs: Additional arguments passed to yaml.dump()
            
        Returns:
            YAML string if stream is None, otherwise None
        """
        # Convert numpy types before dumping
        converted_data = YamlNpEncoder.convert_numpy_types(data)
        
        # Set default YAML dump parameters for better formatting
        default_kwargs = {
            'default_flow_style': False,  # Use block style instead of flow style
            'indent': 2,                  # Use 2-space indentation
            'width': 120,                 # Line width for wrapping
            'allow_unicode': True,        # Allow unicode characters
        }
        
        # Update with user-provided kwargs
        default_kwargs.update(kwargs)
        
        return yaml.dump(converted_data, stream=stream, **default_kwargs)

    @staticmethod
    def dumps(data, **kwargs):
        """
        Serialize data to YAML string with numpy type conversion.
        
        Args:
            data: Data to serialize
            **kwargs: Additional arguments passed to yaml.dump()
            
        Returns:
            YAML string representation
        """
        return YamlNpEncoder.dump(data, stream=None, **kwargs)


class Experiment:
    """
    Base class for all quantum control experiments.
    
    This class provides the fundamental framework for experiment management,
    including configuration loading, data file handling, instrument management,
    and the basic experiment workflow (acquire, analyze, display, save).
    """

    def __init__(
        self,
        path="",
        prefix="data",
        fname=None,
        config_file=None,
        liveplot_enabled=False,
        im=None,
        **kwargs,
    ):
        """
        Initialize the experiment class with configuration and file management.
        
        Args:
            path (str): Directory where data will be stored
            prefix (str): Prefix to use when creating data files (default: "data")
            fname (str, optional): Specific filename to use for data storage
            config_file (str, optional): Configuration file name (relative to path).
                                       If None, looks for path/prefix.json
            liveplot_enabled (bool): Enable live plotting functionality (not implemented)
            im (InstrumentManager, optional): Existing instrument manager instance
            **kwargs: Additional keyword arguments updated to class dictionary
            
        Sets up:
            - File paths and naming
            - Instrument manager for hardware control
            - Configuration loading if config_file specified
        """
        # Update class dictionary with any additional keyword arguments
        self.__dict__.update(kwargs)
        
        # Set up file paths and naming
        self.path = path
        self.prefix = prefix
        self.cfg = None
        
        # Configure config file path
        if config_file is not None:
            self.config_file = os.path.join(path, config_file)
        else:
            self.config_file = None

        # Initialize instrument manager
        if im is None:
            self.im = InstrumentManager()
        else:
            self.im = im

        # Set up data file name
        if fname is not None:
            self.fname = os.path.join(path, fname)
        else:
            # Auto-generate next available filename with .h5 extension
            self.fname = os.path.join(
                path, get_next_filename(path, prefix, suffix=".h5")
            )
            
        # Load configuration if specified
        if config_file is not None:
            self.load_config()

    def load_config(self):
        """
        Load experiment configuration from file.
        
        Supports multiple configuration file formats:
        - .h5 (HDF5) files using SlabFile
        - .yml/.yaml (YAML) files 
        - .json (JSON) files
        
        After loading, creates instrument aliases as class attributes
        based on the "aliases" section in the configuration.
        """
        # Default to prefix.json if no config file specified
        if self.config_file is None:
            self.config_file = os.path.join(self.path, self.prefix + ".json")
            
        try:
            # Handle different config file formats
            if self.config_file[-3:] == ".h5":
                # Load from HDF5 file
                with SlabFile(self.config_file) as f:
                    self.cfg = AttrDict(f.load_config())
                    self.fname = self.config_file
            elif self.config_file[-4:].lower() == ".yml":
                # Load from YAML file
                with open(self.config_file, "r") as fid:
                    self.cfg = AttrDict(yaml.safe_load(fid))
            else:
                # Load from JSON file (default)
                with open(self.config_file, "r") as fid:
                    cfg_str = fid.read()
                    self.cfg = AttrDict(json.loads(cfg_str))

            # Set up instrument aliases as class attributes
            if self.cfg is not None:
                for alias, inst in self.cfg["aliases"].items():
                    if inst in self.im:
                        setattr(self, alias, self.im[inst])
        except Exception as e:
            print("Could not load config.")
            traceback.print_exc()

    def save_config(self):
        """
        Save the current configuration to file and data file attributes.
        
        Saves configuration in two places:
        1. To the config file (if not .h5 format)
        2. To the data file attributes for reproducibility
        """
        # Save to config file if not HDF5
        if self.config_file[:-3] != ".h5":
            with open(self.config_file, "w") as fid:
                json.dump(self.cfg, fid, cls=NpEncoder),
                
        # Always save config to data file attributes
        self.datafile().attrs["config"] = json.dumps(self.cfg, cls=NpEncoder)

    def datafile(self, group=None, remote=False, data_file=None, swmr=False):
        """
        Create and return a SlabFile instance for data storage.
        
        Args:
            group (str, optional): HDF5 group name to create/access within the file
            remote (bool): Remote file access (not currently implemented)
            data_file (str, optional): Specific data file path. Uses self.fname if None
            swmr (bool): Single Writer Multiple Reader mode for concurrent access
                        True: Open in write mode with latest libver for SWMR
                        False: Open in append mode
        
        Returns:
            SlabFile: File handle for data operations
            
        Note:
            Always saves experiment configuration to file attributes for reproducibility.
            Remote proxy functionality is not yet implemented.
        """
        # Use default filename if none specified
        if data_file == None:
            data_file = self.fname
            
        # Configure file access mode based on SWMR setting
        if swmr == True:
            # Write mode with latest libver for SWMR support
            f = SlabFile(data_file, "w", libver="latest")
        elif swmr == False:
            # Append mode for normal operation
            f = SlabFile(data_file, "a")
        else:
            raise Exception("ERROR: swmr must be type boolean")

        # Create/access specific group if requested
        if group is not None:
            f = f.require_group(group)
            
        # Ensure configuration is saved to file attributes
        if "config" not in f.attrs:
            try:
                f.attrs["config"] = json.dumps(self.cfg, cls=NpEncoder)
            except TypeError as err:
                print(("Error in saving cfg into datafile (experiment.py):", err))

        return f

    def go(self, save=False, analyze=False, display=False, progress=False):
        """
        Execute the complete experiment workflow.
        
        This is the main entry point for running experiments, orchestrating
        the data acquisition, analysis, saving, and display steps.
        
        Args:
            save (bool): Whether to save the acquired data
            analyze (bool): Whether to analyze the acquired data
            display (bool): Whether to display/plot the results
            progress (bool): Whether to show progress during acquisition
            
        Workflow:
            1. Acquire data using the acquire() method
            2. Analyze data if requested
            3. Save data if requested  
            4. Display results if requested
        """
        # Step 1: Acquire experimental data
        data = self.acquire(progress)
        
        # Step 2: Analyze data if requested
        if analyze:
            data = self.analyze(data)
            
        # Step 3: Save data if requested
        if save:
            self.save_data(data)
            
        # Step 4: Display results if requested
        if display:
            self.display(data)

    def acquire(self, progress=False, debug=False):
        """
        Acquire experimental data.
        
        This method should be overridden by subclasses to implement
        specific data acquisition procedures for each experiment type.
        
        Args:
            progress (bool): Whether to display progress during acquisition
            debug (bool): Whether to enable debug mode during acquisition
            
        Returns:
            Data acquired from the experiment (format depends on experiment type)
            
        Note:
            Base class implementation is a placeholder that does nothing.
            Subclasses must override this method.
        """
        pass

    def analyze(self, data=None, **kwargs):
        """
        Analyze experimental data.
        
        This method should be overridden by subclasses to implement
        specific analysis procedures for each experiment type.
        
        Args:
            data: Experimental data to analyze (format depends on experiment type)
            **kwargs: Additional analysis parameters
            
        Returns:
            Analyzed data (format depends on experiment type)
            
        Note:
            Base class implementation is a placeholder that does nothing.
            Subclasses must override this method.
        """
        pass

    def display(self, data=None, **kwargs):
        """
        Display or plot experimental results.
        
        This method should be overridden by subclasses to implement
        specific visualization procedures for each experiment type.
        
        Args:
            data: Experimental data to display (format depends on experiment type)
            **kwargs: Additional display parameters
            
        Note:
            Base class implementation is a placeholder that does nothing.
            Subclasses must override this method.
        """
        pass

    def save_data(self, data=None, overwrite=True):
        """
        Save experimental data to file.
        
        Saves a dictionary of data arrays to the experiment's HDF5 file.
        Each key-value pair in the data dictionary becomes a dataset in the file.
        
        Args:
            data (dict, optional): Dictionary containing data to save.
                                 Keys become dataset names, values are converted to numpy arrays.
                                 If None, uses self.data
            overwrite (bool): If True, allows overwriting existing datasets.
                            If False, appends to existing file without overwriting.
        
        Note:
            All data values are converted to numpy arrays before saving.
            This provides a general-purpose save function for dictionary-based data.
            When overwrite=True, opens file in write mode to allow dataset deletion.
        """
        if data is None:
            data = self.data

        # Use write mode if overwrite is True to allow dataset deletion/recreation
        if overwrite:
            with self.datafile(swmr=True) as f:  # swmr=True uses write mode
                for k, d in data.items():
                    f.add(k, np.array(d))
        else:
            # Use append mode for non-overwrite case
            with self.datafile() as f:
                for k, d in data.items():
                    # Check if dataset exists and handle accordingly
                    if k in f:
                        print(f"Warning: Dataset '{k}' already exists. Skipping to avoid overwrite.")
                        continue
                    f.add(k, np.array(d))

    def load_data(self, f):
        """
        Load experimental data from a file handle.
        
        Args:
            f: File handle (typically a SlabFile instance)
            
        Returns:
            dict: Dictionary containing loaded data where:
                  - Keys are dataset names from the file
                  - Values are numpy arrays of the dataset data
                  - Special 'attrs' key contains file attributes dictionary
        """
        data = {}
        
        # Load all datasets as numpy arrays
        for k in f.keys():
            data[k] = np.array(f[k])
            
        # Include file attributes for metadata
        data["attrs"] = f.get_dict()
        
        return data
