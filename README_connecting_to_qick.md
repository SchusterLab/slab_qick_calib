# QICK Board Remote Control Setup with Pyro4

This guide explains how to set up a distributed quantum control system where the QICK board runs on one computer and experiments are controlled remotely from another computer using Pyro4. This is taken from the Schuster Lab wikipage. 

## System Architecture

The system consists of three components:
1. **Nameserver Computer**: Runs the Pyro4 nameserver (directory service)
2. **QICK Board**: QICK board can run python scripts and jupyter notebooks
3. **Experiment Computer**: Connects remotely to run experiments

## Network Requirements

- All computers must be on the same network or have network connectivity
- Firewall must allow connections on the chosen ports (default 9090)
- IP addresses must be known for Nameserver and QICK board.

## Step-by-Step Setup

### Step 1: Start the Nameserver

The nameserver acts as a directory service that allows instruments and clients to find each other. This can be the experiment computer or QICK board or a third computer. 

**Option A: Using the provided notebook**
1. Open `NameServer.ipynb` on the nameserver computer
2. Update the IP address in the script:
   ```python
   # For campus network
   Pyro4.naming.startNSloop(host='10.108.30.23', port=9090)
   
   # For SLAC network  
   Pyro4.naming.startNSloop(host='192.168.137.1', port=9090)
   ```
3. Set the host to your nameserver computer's IP address. If you have issues with procedure, can try changing port number. 
4. Run the notebook cell to start the nameserver

**Option B: Command line**
```bash
python -m Pyro4.naming -n <NAMESERVER_IP> -p 9090
```



**Option C: Python script**
```python
import Pyro4
import Pyro4.naming

def start_nameserver(host_ip, ns_port=9090):
    """Starts a Pyro4 nameserver"""
    Pyro4.config.SERIALIZERS_ACCEPTED = set(['pickle'])
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
    Pyro4.naming.startNSloop(host=host_ip, port=ns_port)

# Replace with your nameserver IP
start_nameserver(host_ip='192.168.1.100', ns_port=9090)
```

It should show the following message when configured correctly:

```
Broadcast server running on 0.0.0.0:9091
NS running on 10.108.30.63:9090 (10.108.30.63)
Warning: HMAC key not set. Anyone can connect to this server!
URI = PYRO:Pyro.NameServer@10.108.30.63:9090
```

### Step 2: Connect QICK Board to Nameserver

On QICK board:

Can go to pyro4 folder to find code start_server. 
This code reads: 
```python 
from qick.pyro import start_server
a=start_server(ns_host="192.168.137.1", ns_port=9090, proxy_name="qick_soc", bitfile='/home/xilinx/jupyter_notebooks/qick_4x2.bit')
```
You choose the proxy name, which will be used when you connect to the Qick board through the instrument server. The bit file may be part of the qick library, or you can download one from here and then upload to the jupyter notebook. https://s3df.slac.stanford.edu/people/meeg/qick/tprocv2/

### Step 3: Connect from Experiment Computer

On the experiment computer where you want to run quantum experiments:

1. **Connect to remote instruments**:
   ```python
   from exp_handling.instrumentmanager import InstrumentManager
   from qick import QickConfig
   
   # Connect to nameserver (client mode - server=False is default)
   im = InstrumentManager(ns_address='<NAMESERVER_IP>', port=9090)
   
   # List available instruments
   print("Available instruments:", list(im.keys()))
   
   # Get the QICK configuration
   soc = QickConfig(im['qick_soc'].get_cfg())
   print("Connected to QICK board:", soc)
   ```
   where you've filled in the name of the qick_soc you chose above. 

2. **Set up experiment configuration**:
   ```python
   import os
   from slab_qick_calib.helpers import config
   
   # Load your experiment configuration
   cfg_file = 'your_config.yml'
   cfg_path = os.path.join('configs', cfg_file)
   auto_cfg = config.load(cfg_path)
   
   # Create experiment dictionary
   expt_path = 'C:\\_Data\\your_experiment\\'
   cfg_dict = {
       'soc': soc, 
       'expt_path': expt_path, 
       'cfg_file': cfg_path, 
       'im': im
   }
   ```

3. **Run experiments** as shown in `tune_qubits_basic.ipynb`:
   ```python
   import slab_qick_calib.experiments as meas
   
   # Example: Run a T1 measurement
   t1 = meas.T1Experiment(cfg_dict, qi=0)
   
   # Example: Run Rabi oscillations
   rabi = meas.RabiExperiment(cfg_dict, qi=0)
   ```

## Configuration Details [all unchecked cline]

### Pyro4 Configuration
The system uses these Pyro4 settings for compatibility:
```python
Pyro4.config.SERIALIZERS_ACCEPTED = set(['pickle'])
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
Pyro4.config.SERVERTYPE = "multiplex"  # Allows concurrent connections
Pyro4.config.REQUIRE_EXPOSE = False    # Simplified object exposure
```

### Network Settings
- **Default nameserver port**: 9090
- **Default Pyro4 timeout**: Auto-configured
- **Serialization**: Pickle (handles complex Python objects)

### IP Address Examples
From the provided files:
- Campus network: `10.108.30.23`
- SLAC network: `192.168.137.1`
- Update these to match your network configuration

## Troubleshooting

### Common Issues

1. **"Could not connect proxies!"**
   - Check nameserver is running: `telnet <NAMESERVER_IP> 9090`
   - Verify IP addresses are correct
   - Check firewall settings

2. **"Name not found in nameserver"**
   - Ensure QICK board computer successfully registered instruments
   - Check instrument server logs for registration messages
   - Verify nameserver is accessible from QICK board computer

3. **Connection timeouts**
   - Check network connectivity between all computers
   - Verify no firewall blocking connections
   - Try increasing Pyro4 timeout: `proxy._pyroTimeout = 30`

4. **Serialization errors**
   - Ensure all computers use same Pyro4 configuration
   - Check Python versions are compatible
   - Verify pickle protocol versions match

### Debugging Commands

**Check nameserver contents**:
```python
import Pyro4
ns = Pyro4.locateNS('<NAMESERVER_IP>', port=9090)
print("Registered objects:", list(ns.list().items()))
```

**Test instrument connection**:
```python
# On experiment computer
proxy = im['qick_soc']
try:
    result = proxy.get_cfg()  # Try calling a method
    print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")
```

**Clean nameserver** (removes stale entries):
```python
im.clean_nameserver()
```

## Security Notes

- This setup uses pickle serialization which can execute arbitrary code
- Only use on trusted networks
- Consider implementing HMAC keys for production use:
  ```python
  Pyro4.config.HMAC_KEY = b'your-secret-key-here'
  ```

## Performance Tips

- Use `multiplex` server type for better concurrent performance
- Set appropriate timeouts based on network latency
- Consider using compression for large data transfers
- Monitor network bandwidth for high-rate experiments

## Example Complete Workflow

1. **Start nameserver** (on dedicated computer or experiment computer):
   ```bash
   python -c "import Pyro4.naming; Pyro4.naming.startNSloop(host='192.168.1.100', port=9090)"
   ```

2. **Start QICK server** (on QICK board computer):
   ```python
   from exp_handling.instrumentmanager import InstrumentManager
   im = InstrumentManager(config_path='qick_instruments.cfg', 
                         server=True, 
                         ns_address='192.168.1.100')
   ```

3. **Connect and run experiments** (on experiment computer):
   ```python
   # Connect
   from exp_handling.instrumentmanager import InstrumentManager
   from qick import QickConfig
   im = InstrumentManager(ns_address='192.168.1.100')
   soc = QickConfig(im['qick_soc'].get_cfg())
   
   # Run experiments
   import slab_qick_calib.experiments as meas
   cfg_dict = {'soc': soc, 'expt_path': './data/', 'cfg_file': 'config.yml', 'im': im}
   t1 = meas.T1Experiment(cfg_dict, qi=0)
   ```

This distributed setup allows for flexible quantum control experiments where the QICK hardware can be isolated while experiments are designed and analyzed on separate computers.
