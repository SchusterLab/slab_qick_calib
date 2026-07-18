#!/usr/bin/env python3
"""
RFSoC Remote Control Server
Runs on the remote computer and provides HTTP API endpoints
for controlling RFSoC bias voltages, filters, and attenuators.
"""

from flask import Flask, request, jsonify
import Pyro4
from qick import QickConfig
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFSoCController:
    def __init__(self, ns_host="192.168.1.36", ns_port=9090, proxy_name="qick_thebe_2"):
        """Initialize connection to RFSoC"""
        self.ns_host = ns_host
        self.ns_port = ns_port
        self.proxy_name = proxy_name
        self.soc = None
        self.soccfg = None
        self.connect()
    
    def connect(self):
        """Establish connection to RFSoC via Pyro4"""
        try:
            Pyro4.config.SERIALIZER = "pickle"
            Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
            
            ns = Pyro4.locateNS(host=self.ns_host, port=self.ns_port)
            self.soc = Pyro4.Proxy(ns.lookup(self.proxy_name))
            self.soccfg = QickConfig(self.soc.get_cfg())
            
            logger.info("Successfully connected to RFSoC")
            logger.info(f"Config: {self.soccfg}")
            
        except Exception as e:
            logger.error(f"Failed to connect to RFSoC: {e}")
            raise
    
    def ensure_connected(self):
        """Ensure connection is still active, reconnect if needed"""
        try:
            # Test connection with a simple call
            self.soc.get_cfg()
        except:
            logger.warning("Connection lost, attempting to reconnect...")
            self.connect()
    
    def set_gen_filter(self, channel, fc, ftype):
        """Set generator filter parameters"""
        self.ensure_connected()
        return self.soc.rfb_set_gen_filter(channel, fc=fc, ftype=ftype)
    
    def set_ro_filter(self, channel, fc, ftype):
        """Set readout filter parameters"""
        self.ensure_connected()
        return self.soc.rfb_set_ro_filter(channel, fc=fc, ftype=ftype)
    
    def set_gen_rf(self, channel, first_stage_atten, second_stage_atten):
        """Set generator RF attenuation"""
        self.ensure_connected()
        return self.soc.rfb_set_gen_rf(channel, first_stage_atten, second_stage_atten)
    
    def set_ro_rf(self, channel, atten):
        """Set readout RF attenuation"""
        self.ensure_connected()
        return self.soc.rfb_set_ro_rf(channel, atten)
    
    def get_bias(self, bias_ch):
        """Read bias voltage setpoint"""
        self.ensure_connected()
        return self.soc.rfb_get_bias(bias_ch)
    
    def set_bias(self, bias_ch, voltage):
        """Set bias voltage"""
        self.ensure_connected()
        return self.soc.rfb_set_bias(bias_ch, voltage)

# Initialize Flask app and RFSoC controller
app = Flask(__name__)
controller = RFSoCController()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "RFSoC server is running"})

@app.route('/gen_filter', methods=['POST'])
def set_gen_filter():
    """Set generator filter endpoint"""
    try:
        data = request.json
        channel = data['channel']
        fc = data['fc']
        ftype = data['ftype']
        
        if ftype not in ['bandpass', 'lowpass', 'highpass', 'bypass']:
            return jsonify({"error": "Invalid filter type"}), 400
        
        result = controller.set_gen_filter(channel, fc, ftype)
        return jsonify({"result": result, "status": "success"})
    
    except Exception as e:
        logger.error(f"Error in set_gen_filter: {e}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/ro_filter', methods=['POST'])
def set_ro_filter():
    """Set readout filter endpoint"""
    try:
        data = request.json
        channel = data['channel']
        fc = data['fc']
        ftype = data['ftype']
        
        if ftype not in ['bandpass', 'lowpass', 'highpass', 'bypass']:
            return jsonify({"error": "Invalid filter type"}), 400
        
        result = controller.set_ro_filter(channel, fc, ftype)
        return jsonify({"result": result, "status": "success"})
    
    except Exception as e:
        logger.error(f"Error in set_ro_filter: {e}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/gen_rf', methods=['POST'])
def set_gen_rf():
    """Set generator RF attenuation endpoint"""
    try:
        data = request.json
        channel = data['channel']
        first_stage = data['first_stage_atten']
        second_stage = data['second_stage_atten']
        
        result = controller.set_gen_rf(channel, first_stage, second_stage)
        return jsonify({"result": result, "status": "success"})
    
    except Exception as e:
        logger.error(f"Error in set_gen_rf: {e}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/ro_rf', methods=['POST'])
def set_ro_rf():
    """Set readout RF attenuation endpoint"""
    try:
        data = request.json
        channel = data['channel']
        atten = data['atten']
        
        result = controller.set_ro_rf(channel, atten)
        return jsonify({"result": result, "status": "success"})
    
    except Exception as e:
        logger.error(f"Error in set_ro_rf: {e}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/bias', methods=['GET', 'POST'])
def bias():
    """Get or set bias voltage"""
    try:
        if request.method == 'GET':
            bias_ch = int(request.args.get('channel'))
            voltage = controller.get_bias(bias_ch)
            return jsonify({"voltage": voltage, "channel": bias_ch, "status": "success"})
        
        elif request.method == 'POST':
            data = request.json
            bias_ch = data['channel']
            voltage = data['voltage']
            
            result = controller.set_bias(bias_ch, voltage)
            return jsonify({"result": result, "status": "success"})
    
    except Exception as e:
        logger.error(f"Error in bias endpoint: {e}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == '__main__':
    print("Starting RFSoC Remote Control Server...")
    print("Available endpoints:")
    print("  POST /gen_filter - Set generator filter")
    print("  POST /ro_filter - Set readout filter")
    print("  POST /gen_rf - Set generator RF attenuation")
    print("  POST /ro_rf - Set readout RF attenuation")
    print("  GET/POST /bias - Get/set bias voltage")
    print("  GET /health - Health check")
    
    # Run the server
    # Use host='0.0.0.0' to allow external connections
    app.run(host='0.0.0.0', port=5000, debug=False)