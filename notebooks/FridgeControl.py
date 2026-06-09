"""
Fridge controller functions from Zoe

IP is hard-coded for BFG

"""

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from time import sleep


host = 'https://192.168.137.51'
port = '49098'
key = '98b3c002-4643-4bef-b209-870b221a2cef'


def getVarMap(var):
    return 'mapper.'+var.replace('/','.')

def getData(var):
    path = '/values/mapper/'+var
    dataRaw = requests.get(host+':'+port+path,params = {'key':key}, verify=False)
    data = dataRaw.json()
    return data

def getValue(var):
    data = getData(var)
    return data['data'][getVarMap(var)]['content']['latest_value']['value']

def setMXCHeater(value):
    powerStr = str(value*1e-6)

    content = {
        "data" : {
            "mapper.temperature_control.heaters.sample.power" : {
            "content" : {"value" : powerStr}}}}

    path='/values/'
    dataRaw = requests.post(host+':'+port+path,params = {'key':key}, verify=False, json=content)
    data = dataRaw.json()
    return data

def getMXCHeaterPower():
    return float(getValue('temperature_control/heaters/sample/power'))*1e6

def getMXCTemperature():
    mxc_tempval = getValue('bf/temperatures/tmixing')
    if mxc_tempval == '':
        mxc_tempval = 0
    return float(mxc_tempval)*1e3

def setStillHeater(value):
    powerStr = str(value*1e-3)

    content = {
        "data" : {
            "mapper.temperature_control.heaters.still.power" : {
            "content" : {"value" : powerStr}}}}

    path='/values/'
    dataRaw = requests.post(host+':'+port+path,params = {'key':key}, verify=False, json=content)
    data = dataRaw.json()
    return data

def getStillHeaterPower():
    return float(getValue('temperature_control/heaters/still/power'))*1e3

def getStillTemperature():
    return float(getValue('bf/temperatures/tstill'))*1e3

def getP3Pressure():
    # returns P3 pressure in mBar
    return float(getValue('bf/pressures/p3'))*1e3