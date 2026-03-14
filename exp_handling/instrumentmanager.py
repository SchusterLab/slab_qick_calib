# -*- coding: utf-8 -*-
"""
Pyro4-based instrument manager for remote QICK hardware access.

Connects to a Pyro4 nameserver to create proxies for all registered
instruments, enabling remote control from notebooks and scripts.

Original code by David Schuster, from the slab repository.
"""

import Pyro4

# Pyro4 global configuration
Pyro4.config.SERVERTYPE = "multiplex"
Pyro4.config.REQUIRE_EXPOSE = False
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = set(["json", "marshal", "serpent", "pickle"])


class InstrumentManager(dict):
    """Dict-based container for remote instruments accessed via Pyro4 proxies.

    Typical usage in notebooks::

        im = InstrumentManager(ns_address='192.168.1.100', port=8888)
        soc = im['soc']          # dict-style access
        soc = im.soc             # attribute-style access

    When called with no arguments (as in Experiment.__init__), creates an
    empty dict without attempting any network connection.

    Args:
        ns_address: Pyro4 nameserver IP address. None skips connection.
        port: Nameserver port (default 9090).
    """

    def __init__(self, ns_address=None, port=9090):
        dict.__init__(self)
        self.ns_address = ns_address
        self.port = port
        if ns_address is not None:
            try:
                self.connect_proxies()
            except (Pyro4.errors.NamingError, Pyro4.errors.CommunicationError,
                    ConnectionRefusedError, TimeoutError, OSError) as e:
                print(f"Warning: Could not connect to nameserver "
                      f"at {ns_address}:{port} -- {e}")

    def connect_proxies(self):
        """Connect to all instruments registered in the Pyro4 nameserver."""
        ns = Pyro4.locateNS(self.ns_address, port=self.port)
        for name, uri in list(ns.list().items()):
            self[name] = Pyro4.Proxy(uri)

    def __getattr__(self, item):
        """Allow attribute-style access to instruments (im.soc)."""
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)
