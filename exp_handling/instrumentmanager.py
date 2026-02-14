# -*- coding: utf-8 -*-
"""
Instrument management system for quantum control experiments.

This module provides the InstrumentManager class for managing laboratory instruments
using Pyro4 for distributed object communication. It handles instrument configuration,
proxy connections, and server functionality.

Original code by David Schuster
Taken from the slab repository

Created on Sat Sep 03 14:50:09 2011
@author: David Schuster
"""
from pathlib import Path
import socket
import sys
from optparse import OptionParser
import Pyro4

Pyro4Loaded = True
# Block calls from running simultaneously
Pyro4.config.SERVERTYPE = "multiplex"
Pyro4.config.REQUIRE_EXPOSE = False
# Pyro4.config.HMAC_KEY = b'6551d449b0564585a9d39c0bd327dcf1'
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = set(["json", "marshal", "serpent", "pickle"])
# except ImportError:
#     print("Warning: Pyro4 package is not present")
#     print("Instrument Servers will not work.")
#     Pyro4Loaded = False


class InstrumentManager(dict):
    """
    InstrumentManager class for managing laboratory instruments via Pyro4.
    
    This class extends dict to provide a container for instruments that can be
    accessed by name. It handles both client and server modes:
    - Client mode: Connects to remote instruments via Pyro4 proxies
    - Server mode: Serves local instruments to remote clients
    
    The class reads configuration files to determine which instruments to load
    and their connection parameters.
    
    Args:
        config_path (str, optional): Path to instrument configuration file
        server (bool): If True, run as instrument server; if False, run as client
        ns_address (str, optional): Address of Pyro4 nameserver
        port (int): Port number for nameserver connection (default: 9090)
    """

    def __init__(self, config_path=None, server=False, ns_address=None, port=9090):
        """
        Initialize InstrumentManager in either client or server mode.
        
        In client mode, attempts to connect to remote instruments via Pyro4 proxies.
        In server mode, loads instruments from config and serves them to clients.
        
        Args:
            config_path (str, optional): Path to configuration file
            server (bool): Server mode if True, client mode if False
            ns_address (str, optional): Nameserver address for Pyro4
            port (int): Nameserver port (default: 9090)
        """
        dict.__init__(self)
        
        # Store configuration parameters
        self.config_path = config_path
        self.config = None
        self.ns_address = ns_address
        self.port = port
        
        # Client mode: Connect to remote instruments via proxies
        if not server and Pyro4Loaded:
            try:
                # self.clean_nameserver()  # Optional cleanup
                self.connect_proxies()
            except Exception as e:
                print("Warning: Could not connect proxies!")
                print(e)
        
        # Load instruments from configuration file
        if config_path is not None:
            instruments = self.load_config_file(config_path)
        else:
            instruments = []
        
        # Server mode: Serve instruments to remote clients
        if server and Pyro4Loaded:
            self.serve_instruments(instruments)
        else:
            # Client mode: Add instruments to local dictionary
            for instrument in instruments:
                self[instrument.name] = instrument

    def line_is_comment_or_empty(self, line=""):
        """
        Check if a configuration line is a comment or empty.
        
        Args:
            line (str): Line from configuration file
            
        Returns:
            bool: True if line is comment (starts with #) or empty
        """
        _line = line.strip()
        if len(_line) == 0 or _line[0] == "#":
            return True
        else:
            return False

    def parse_config_string(self, line):
        """
        Parse a configuration line to extract instrument parameters.
        
        Expected format: "name instrument_class address"
        
        Args:
            line (str): Configuration line
            
        Returns:
            tuple: (name, instrument_class, address)
        """
        params = line.split()
        name, instrument_class, address = params
        return name, instrument_class, address

    def load_config_file(self, config_path):
        """
        Load instrument configuration from file.
        
        Parses configuration file line by line, skipping comments and empty lines.
        Each valid line should contain: name instrument_class address
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            list: List of loaded instrument instances
        """
        print("Loaded Instruments: ", end=" ")
        f = open(config_path, "r")
        instruments = []
        
        for line in f.readlines():
            isComment = self.line_is_comment_or_empty(line)
            if not isComment:
                name = self.parse_config_string(line)[0]
                instruments.append(self.load_instrument(line))
                
        print("!")
        return instruments

    def load_instrument(self, config_string):
        """
        Load a single instrument based on configuration string.
        
        Creates an instrument instance by calling the appropriate class
        from slab.instruments with the specified name and address.
        
        Args:
            config_string (str): Configuration line (name class address)
            
        Returns:
            Instrument instance
        """
        name, in_class, addr = self.parse_config_string(config_string)
        fn = getattr(slab.instruments, in_class)
        return fn(name=name, address=addr)

    def __getattr__(self, item):
        """
        Enable attribute-style access to instruments.
        
        Allows accessing instruments as attributes (im.instrument_name) 
        instead of dictionary keys (im['instrument_name']).
        Only called if there isn't a regular attribute with this name.
        
        Args:
            item (str): Instrument name
            
        Returns:
            Instrument instance
            
        Raises:
            AttributeError: If instrument not found
        """
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    def set_alias(self, name, alias):
        """
        Create an alias for an instrument.
        
        Allows accessing the same instrument by multiple names.
        
        Args:
            name (str): Existing instrument name
            alias (str): New alias name
        """
        self[alias] = self[name]

    def serve_instruments(self, instruments=None):
        """
        Start Pyro4 daemon to serve instruments to remote clients.
        
        Creates a Pyro4 daemon, registers all instruments with the nameserver,
        and starts the request loop to handle remote calls. This method blocks
        until the server is shut down.
        
        Args:
            instruments (list): List of instrument instances to serve
            
        Note:
            This method will run indefinitely, serving instrument requests.
            Supports autoproxy for exposing instrument properties as separate objects.
        """
        Pyro4.config.SERVERTYPE = "multiplex"
        
        # Get local IP address by connecting to external address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        host = s.getsockname()[0]
        
        # Create daemon and locate nameserver
        daemon = Pyro4.Daemon(host=host)
        ns = Pyro4.locateNS(self.ns_address)

        # Register each instrument with the daemon and nameserver
        for instrument in instruments:
            uri = daemon.register(instrument)
            ns.register(instrument.name, uri)

            # Register autoproxy objects if supported
            # See: https://pyro4.readthedocs.io/en/stable/servercode.html#autoproxying
            if hasattr(instrument, "autoproxy"):
                for obj in instrument.autoproxy:
                    daemon.register(obj)

            print("Registered: %s\t%s" % (instrument.name, uri))
            
        # Start serving requests (blocks indefinitely)
        daemon.requestLoop()

    def connect_proxies(self):
        """
        Connect to all instruments listed in the Pyro4 nameserver.
        
        Queries the nameserver for all registered instruments and creates
        Pyro4 proxy objects for each one, adding them to the local dictionary.
        This enables remote access to instruments running on other machines.
        """
        ns = Pyro4.locateNS(self.ns_address, port=self.port)
        
        # Create proxy for each instrument in nameserver
        for name, uri in list(ns.list().items()):
            self[name] = Pyro4.Proxy(uri)

    def get_settings(self):
        """
        Retrieve current settings from all managed instruments.
        
        Calls get_settings() on each instrument in the manager and collects
        the results. Handles exceptions gracefully if any instrument fails.
        
        Returns:
            list: List of instrument settings dictionaries
        """
        settings = []
        for k, inst in self.items():
            try:
                settings.append(inst.get_settings())
            except:
                print("Warning! Could not get settings for instrument: %s" % k)
        return settings

    def save_settings(self, path, prefix=None, params={}):
        """
        Save current settings from all instruments to a configuration file.
        
        Retrieves settings from all instruments and saves them to a .cfg file
        for later reference or restoration. Useful for documenting experiment
        conditions and instrument states.
        
        Args:
            path (str): Directory path or full file path for settings file
            prefix (str, optional): Filename prefix if path is directory
            params (dict): Additional parameters to include in settings file
        """
        settings = self.get_settings()
        settings.append(params)
        
        # Construct filename
        if prefix:
            fname = str(Path(path) / prefix)
        else:
            fname = path
            
        # Ensure .cfg extension
        if ".cfg" not in fname.lower():
            fname += ".cfg"
            
        # Write settings to file
        f = open(fname, "w")
        for s in settings:
            f.write(repr(s))
            f.write("\n")
        f.close()

    def clean_nameserver(self):
        """
        Clean up stale entries from the Pyro4 nameserver.
        
        Checks each registered instrument in the nameserver to verify it's
        still responsive. Removes entries for instruments that are no longer
        accessible (e.g., due to network issues or server shutdown).
        
        This helps maintain a clean nameserver registry by removing dead entries.
        """
        ns = Pyro4.locateNS(self.ns_address)
        
        # Test each registered instrument
        for name, uri in list(ns.list().items()):
            try:
                proxy = Pyro4.Proxy(uri)
                proxy._pyroTimeout = 0.1  # Short timeout for responsiveness check
                proxy.get_id()  # Test call to verify instrument is responsive
            except:
                # Remove unresponsive instruments from nameserver
                ns.remove(name)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    
    parser = OptionParser()
    parser.add_option(
        "-f", "--file", dest="filename", help="Config file to load", metavar="FILE"
    )
    parser.add_option(
        "-s",
        "--server",
        action="store_true",
        dest="server",
        default=False,
        help="Act as instrument server",
    )
    parser.add_option(
        "-n",
        "--nameserver",
        "--ns_address",
        action="store",
        type="string",
        dest="ns_address",
        help="Address of name server (default auto-lookup)",
    )
    parser.add_option(
        "-g",
        "--gui",
        action="store_true",
        dest="gui",
        default=False,
        help="Run Instrument Manager in gui mode",
    )
    parser.add_option(
        "-i",
        action="store_true",
        dest="interact",
        default=False,
        help="interactive option not used.",
    )
    options, args = parser.parse_args(args)

    if options.gui:
        sys.exit(
            slab.gui.runWin(
                InstrumentManagerWindow,
                filename=options.filename,
                nameserver=options.ns_address,
            )
        )
    else:
        im = InstrumentManager(
            config_path=options.filename,
            server=options.server,
            ns_address=options.ns_address,
        )
        globals().update(im)
        globals()["im"] = im
        try:
            globals()["plotter"] = liveplot.LivePlotClient()
        except:
            print("Warning: Couldn't load liveplotter")


if __name__ == "__main__":
    try:
        import slab.gui
        from slab.instruments import InstrumentManagerWindow
    except:
        print("Warning: Could not import slab.gui or InstrumentManagerWindow!")
    try:
        import liveplot
    except:
        print("Warning: Could not load liveplot")

    main(sys.argv[1:])
