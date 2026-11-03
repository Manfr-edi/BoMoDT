# ****************************************************
# Module Purpose:
#   This library defines the Agent class, which is responsible for interacting with the FIWARE IoT Agent.
#   The Agent class manages the registration of devices and service groups, and handles the sending of
#   measurements to the IoT Agent.
#
#   - The Agent class includes attributes for agent identification, such as agent ID, port numbers,
#     FIWARE service, and service path.
#   - It provides methods for checking if a service group or device is already registered, registering
#     service groups, devices, and measurements, and sending data to the IoT Agent.
#   - The Agent class also handles the retrieval and transmission of sensor data, ensuring that measurements
#     are correctly formatted and sent to the specified endpoints.
#
# ****************************************************

import json
import requests
from requests.structures import CaseInsensitiveDict
from libraries.classes.Broker import *
import time
from uuid import uuid4
from libraries.utils.executionTiming import record_send

class Agent:
    agentID: str
    hostname: str
    brokerPortNumber: int
    southPortNumber: int
    northPortNumber: int
    fiwareService: str
    fiwareServicePath: str
    cbReference: Optional['Broker']
    cbConnection: Optional[Client]

    def __init__(self, aid: str, hostname: str, cb_port: int, south_port: int, northport: int, fw_service: str, fw_path: str):
        self.agentID = aid
        self.hostname = hostname
        self.brokerPortNumber = cb_port
        self.southPortNumber = south_port
        self.northPortNumber = northport
        self.fiwareService = fw_service
        self.fiwareServicePath = fw_path
        self.cbReference = None
        self.cbConnection = None
        self.timing_session_id = str(uuid4())

    def isServiceGroupRegistered(self, entity_type: str):
        # Check if the device is already registered
        url = "http://{}:{}/iot/services".format(self.hostname,self.northPortNumber)
        header = CaseInsensitiveDict()
        header["fiware-service"] = self.fiwareService
        header["fiware-servicepath"] = self.fiwareServicePath
        response = requests.get(url, headers=header, timeout=15)
        # TODO: improve the type of check done to find the service group
        if entity_type in response.text:
            # print("Entity type found inside the body response")
            return True
        else:
            return False

    def isDeviceRegistered(self, device_id: str):
        # Check if the device is already registered
        url = "http://{}:{}/iot/devices/{}".format(self.hostname, self.northPortNumber, device_id)
        header = CaseInsensitiveDict()
        header["fiware-service"] = self.fiwareService
        header["fiware-servicepath"] = self.fiwareServicePath
        response = requests.get(url, headers=header, timeout=15)
        # TODO: improve the type of check done to find the device
        if response.status_code == 200:
            # print("Device found")
            return True
        else:
            return False

    def getServiceGroupKey(self, entity_type):
        serviceKey = None

        # Check if the device is already registered
        url = "http://{}:{}/iot/services".format(self.hostname, self.northPortNumber)
        header = CaseInsensitiveDict()
        header["fiware-service"] = self.fiwareService
        header["fiware-servicepath"] = self.fiwareServicePath
        response = requests.get(url, headers=header, timeout=15)
        # TODO: improve the type of check done to find the key
        if entity_type in response.text:
            # print("Entity type found inside the body response")
            services = response.json()["services"]
            for serv in services:
                if serv["entity_type"] == entity_type:
                    # print("Found it!")
                    serviceKey = serv["apikey"]
                    break
        return serviceKey

    def serviceGroupRegistration(self, api_key: str, entity_type: str):
        # register a Service Group in the IoT Agent (JSON version)
        if self.isServiceGroupRegistered(entity_type):
            print("Service is already registered. Skipping registration.")
            return True

        url_registration = "http://{}:{}/iot/services".format(self.hostname, self.northPortNumber)
        # building packet header and payload
        header = CaseInsensitiveDict()
        header["Content-Type"] = "application/json"
        header["fiware-service"] = self.fiwareService
        header["fiware-servicepath"] = self.fiwareServicePath

        payload = {
            "services": [
                {
                    "apikey": api_key,
                    "cbroker": "http://orion:{}".format(self.brokerPortNumber),
                    "entity_type": entity_type,
                    "resource": "/iot/json"
                }
            ]
        }
        reg_response = requests.post(url_registration, headers=header, data=json.dumps(payload), timeout=15)
        return reg_response

    def measurementRegistration(self, measure_type: str, device_id: str, entity_type: str, timezone,
                                controlled_asset: str):
        if self.isDeviceRegistered(device_id):
            # Upgrade existing registrations without deleting their attributes.
            url = f"http://{self.hostname}:{self.northPortNumber}/iot/devices/{device_id}"
            headers = {'fiware-service': self.fiwareService, 'fiware-servicepath': self.fiwareServicePath}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            attributes = response.json().get('attributes', [])
            missing = [name for name in ('timingId', 'timingSession')
                       if not any(attribute.get('name') == name for attribute in attributes)]
            if missing:
                attributes.extend({'object_id': name, 'name': name, 'type': 'Text'} for name in missing)
                response = requests.put(url, headers=headers, json={'attributes': attributes}, timeout=10)
                response.raise_for_status()
            print("Device is already registered. Skipping registration.")
            return True
        url_registration = "http://{}:{}/iot/devices".format(self.hostname, self.northPortNumber)
        # building packet header and payload
        header = CaseInsensitiveDict()
        header["Content-Type"] = "application/json"
        header["fiware-service"] = self.fiwareService
        header["fiware-servicepath"] = self.fiwareServicePath

        payload = {}
        if measure_type == "trafficFlow":
            payload = {
                "devices": [
                    {
                        "device_id": device_id,
                        "entity_name": "urn:ngsi-ld:Device:{}".format(device_id),
                        "entity_type": entity_type,
                        "timezone": timezone,
                        "attributes": [
                            {"object_id": "trafficFlow", "name": "trafficFlow", "type": "Integer"},
                            {"object_id": "location", "name": "location", "type": "GeoProperty"}, #GeoProperty is correct
                            {"object_id": "laneDirection", "name": "laneDirection", "type": "TextUnrestricted"},
                            {"object_id": "timeSlot", "name": "timeSlot", "type": "TextUnrestricted"},
                            {"object_id": "dateObserved", "name": "dateObserved", "type": "TextUnrestricted"},
                            {"object_id": "timingId", "name": "timingId", "type": "Text"},
                            {"object_id": "timingSession", "name": "timingSession", "type": "Text"}
                        ],
                        "static_attributes": [
                            {"name": "deviceCategory", "type": "Text", "value": "Sensor"},
                            {"name": "controlledAsset", "type": "Relationship", "value": controlled_asset}
                        ]
                    }]}
        # print(payload)
        reg_response = requests.post(url_registration, headers=header, data=json.dumps(payload), timeout=15)
        return reg_response

    def retrievingData(self, *data, device_id, device_key):
        # TODO: befor even retrieving data we should check if there exist a device with
        #  that ID and if for that device the provided key is correct.
        date = data[0][0]
        timeSlot = data[0][1]
        flow = data[0][2]
        coordinates = data[0][3]
        direction = data[0][4]
        taz = data[0][5]
        self.measurementSending(date, timeSlot, flow, coordinates, direction, taz, measure_type="trafficFlow", device_key=device_key,
                                device_id=device_id)

    def measurementSending(self, date: str, timeSlot: str, flow: int, coordinates, direction: str, taz: str, measure_type: str, device_key, device_id):
        """
        Send a traffic flow measurement to the IoT Agent and update the Context Broker.

        :param date: Date of the observation in day/month/year format
        :param timeSlot: Time slot of the observation (e.g., "08:00-09:00") on 24h format.
        :param flow: Measured traffic flow.
        :param coordinates: GPS coordinates (longitude, latitude) of the traffic loop.
        :param direction: Direction of the lane (e.g., "N", "S", "E", "W").
        :param measure_type: Type of the measurement (e.g., "trafficFlow").
        :param device_key: API key for the device.
        :param device_id: ID of the device.

        :returns: True if the measurement is sent and context is updated successfully, False otherwise.
        :raises: Exception if any failure occurs during the process.
        """


        url_sending = "http://{}:{}/iot/json?k={}&i={}".format(self.hostname, self.southPortNumber, device_key, device_id)
        # building packet header and payload
        header = CaseInsensitiveDict()
        header["Content-Type"] = "application/json"
        header["fiware-service"] = "openiot"
        header["fiware-servicepath"] = "/"

        payload = {}
        if measure_type == "trafficFlow":
            payload = {
                "trafficFlow": flow,
                "location": {
                    "type": "Point",
                    "coordinates": coordinates
                },
                "taz": taz,
                "timeSlot": timeSlot,
                "laneDirection": direction,
                "dateObserved": date
            }

        try:
            payload['timingId'] = str(uuid4())
            payload['timingSession'] = self.timing_session_id
            self.last_measurement_id = payload['timingId']
            encoded_payload = json.dumps(payload)
            sent_at_ms = time.time() * 1000
            try:
                sendingResponse = requests.post(url_sending, headers=header, data=encoded_payload, timeout=15)
            finally:
                record_send(self.timing_session_id, f'urn:ngsi-ld:Device:{device_id}',
                            date, timeSlot, sent_at_ms, payload['timingId'])
            sendingResponse.raise_for_status()  # Raise an error if the status code is not 2xx
        except requests.RequestException as e:
            raise Exception(f"Failed to send data to IoT Agent: {e}")

        if sendingResponse.status_code == 200 or sendingResponse.status_code == 201:
            print("Data sent successfully to IoT Agent!")
            # time.sleep(3)
            try:
                if self.cbReference is None:
                    self.cbReference = Broker(pn=self.brokerPortNumber, pnt=None, host=self.hostname, fiwareservice=self.fiwareService)
                if self.cbConnection is None:
                    self.cbConnection = self.cbReference.createConnection()
                print("Updating Context Broker entities linked to device: " + str(device_id) + "...")
                # time.sleep(3)
                cbResponse = self.cbReference.updateContext(deviceID=device_id, date=date, timeSlot=timeSlot, trafficFlow=flow,
                                                  coordinates=coordinates,laneDirection=direction, taz=taz, cbConnection=self.cbConnection,
                                                  timing_id=payload['timingId'], timing_session=self.timing_session_id)
                if cbResponse is True:
                    return True
                else:
                    raise Exception(f"Context update failed for device {device_id}.")

            except Exception as e:
                raise Exception(f"Failed to update context broker: {e}")
        else:
            raise Exception(f"Failed to send measurement. HTTP status code: {sendingResponse.status_code}")
