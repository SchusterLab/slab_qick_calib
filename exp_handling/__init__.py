"""
Experiment handling for quantum control experiments.

Core components:
    Experiment          - Base class for experiment orchestration
    NpEncoder           - JSON encoder for numpy types
    YamlNpEncoder       - YAML encoder for numpy types
    SlabFile            - HDF5 data storage wrapper
    AttrDict            - Dictionary with attribute-style access
    InstrumentManager   - Pyro4-based remote instrument access
"""

from .experiment import Experiment, NpEncoder, YamlNpEncoder
from .datamanagement import SlabFile, AttrDict
from .instrumentmanager import InstrumentManager
