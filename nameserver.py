"""Start the Pyro4 nameserver (port 8888) that QICK boards and experiment PCs register with."""

import Pyro4
import Pyro4.naming

def start_nameserver(host_ip, ns_port=9090):
    """Starts a Pyro4 nameserver"""
    Pyro4.config.SERIALIZERS_ACCEPTED = set(['pickle'])
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
    Pyro4.naming.startNSloop(host=host_ip, port=ns_port)

# Replace with your nameserver IP
#ip = '192.168.137.1'
ip = '10.108.30.23'
start_nameserver(host_ip=ip, ns_port=8888)