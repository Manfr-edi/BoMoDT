import os
import pandas as pd
import string
import secrets
import time
from datetime import datetime, timedelta
import random

from bson import ObjectId
from pymongo import MongoClient

from libraries.classes.TrafficModeler import TrafficModeler
from libraries.constants import SUMO_PATH, SUMO_NET_FILE_PATH, PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH
from libraries.classes.SumoSimulator import Simulator


def readingFiles(folder: str):
    """
    Read all CSV files in the specified folder, using the file names as dictionary keys.

    Each CSV file is read with `;` as the separator, and the data is stored in a dictionary with
    the file names (without extensions) as keys. It also returns a list of these file names.

    :param folder: Path to the folder containing CSV files.
    :return: Tuple of two elements:
             - data (dict): Dictionary where each key is a file name (without extension), and each value is the data
               from the corresponding CSV file as a DataFrame.
             - files (list): List of file names (without extensions) for the loaded CSV files.
    """
    files = os.listdir(folder)
    data = {}

    for i, file in enumerate(files):
        file_path = os.path.join(folder, file)
        key = os.path.splitext(file)[0]  # Remove .csv extension to use file name as dictionary key
        data[key] = pd.read_csv(file_path, sep=';')  # Read CSV with ; as the separator
        files[i] = key  # Replace the original file name in `files` with the name without extension

    return data, files

def generate_random_key(length):
    """
    Generate a random alphanumeric key with a given length
    Args:
        length: the length of the key that will be generated

    Returns:
        the alphanumeric random key string generated

    """
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for i in range(length))

def loadEnvVar(file_path):
    env_vars = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # remove whitespace
            if line and not line.startswith('#'):  # line is not a comment
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()  # removing leading or trailing spaces
    return env_vars

# function to calculate difference between actual and previous date time of dataset entries to properly simulate devices
def convert_float(inp):
    splitted_data = inp.split(",")
    return float(splitted_data[-2]), float(splitted_data[-1])


# COMMENTED PARTS ARE FOR TIME ESTIMATION PURPOSES
def processingTlData(timeSlot, trafficData, roads: dict):
    timestamps = []
    client = MongoClient("mongodb://localhost:27017/")
    db = client["orion-openiot"]
    collection = db["entities"]
    for index, row in trafficData.iterrows():
        trafficFlow = row["flow"]
        raw_coordinates = row["geopoint"]
        longitude, latitude = map(float, raw_coordinates.split(','))
        coordinates=[longitude,latitude]
        direction = str(row["direction"])
        roadName = row['road_name']
        date = row['date']
        taz = row['taz']

        if roadName in roads:
            trafficLoopIdentifier= "TL{}".format(str(row["ID_loop"]))
            trafficLoopSensor = roads[roadName].getSensor(trafficLoopIdentifier)
            partial_id = trafficLoopSensor.devicePartialID
            entry = collection.find_one({"_id.id": {"$regex": partial_id}})
            # if index == len(trafficData) - 1:
            #     old_mod_date = entry["modDate"] if entry else None


            if trafficLoopSensor is not None and trafficLoopSensor.name == "TL":
                # if index == 0:
                #     first_start_time = time.time_ns() / 1e9
                #
                # start_time = time.time_ns()
                # print("Sending new traffic loop measurement to IoT agent...")
                # print("ID Traffic Loop Device: " + str(trafficLoopSensor.devicePartialID))
                # print("Traffic Flow Measurment: " + str(trafficFlow))
                # print("Traffic Loop Position: [" + str(longitude) + ", " + str(latitude) + "]")
                # print("Road: " + str(roadName))

                trafficLoopSensor.sendData(date, timeSlot, trafficFlow, coordinates, direction, taz,
                                           device_id=trafficLoopSensor.devicePartialID,
                                           device_key=trafficLoopSensor.apiKey)
                # end_time = time.time_ns()
                # if index == len(trafficData) - 1:
                #     new_mod_date = wait_for_mod_date_change(entry_id=partial_id, old_mod_date=old_mod_date)
                #     elapsed_time = (new_mod_date-first_start_time)
                #     elapsed_time = f"{int(elapsed_time)},{str(round(elapsed_time % 1, 4))[2:]}"
                #     timestamps.append({"evento": "One Hour Measurement", "Sensor TL": "All","start_timestamp": first_start_time, "end_timestamp": new_mod_date,
                #                        "elapsed_time": elapsed_time})
                # else:
                #     timestamps.append({"evento": "Sending Data", "Sensor TL": str(trafficLoopSensor.devicePartialID),"start_timestamp": start_time, "end_timestamp": end_time,
                #                     "elapsed_time": end_time-start_time})
    # configurationPath = SUMO_PATH + "/standalone"
    # logFile = "./command_log.txt"
    # sumoSimulator = Simulator(configurationPath=configurationPath, logFile=logFile)
    # print("Instantiating a Traffic Modeler...")
    # basemodel = TrafficModeler(simulator=sumoSimulator, trafficDataFile=PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH,
    #                            sumoNetFile=SUMO_NET_FILE_PATH,
    #                            date='2024-02-01',
    #                            timeSlot=timeSlot,
    #                            modelType="greenshield")
    # basemodel.saveTrafficData(outputDataPath="time/tm_time/model.csv")
    # modeled_time = time.time_ns() / 1e9
    # basemodel.vTypeGeneration(modelType="W99", tau="1.5",
    #                                                    additionalParam={"cc1": "1.3", "cc2": "8"})
    # basemodel.runSimulation(withGui=False)
    # simulated_time = time.time_ns() / 1e9
    # elapsed_modeling_time = (modeled_time - first_start_time)
    # elapsed_modeling_time = f"{int(elapsed_modeling_time)},{str(round(elapsed_modeling_time % 1, 4))[2:]}"
    # elapsed_simulating_time = (simulated_time - first_start_time)
    # elapsed_simulating_time = f"{int(elapsed_simulating_time)},{str(round(elapsed_simulating_time % 1, 4))[2:]}"
    # timestamps.append(
    #     {"evento": "Sending Data", "Sensor TL": str(trafficLoopSensor.devicePartialID), "start_timestamp": start_time,
    #      "end_timestamp": end_time,
    #      "elapsed_time": end_time - start_time})
    # timestamps.append({"evento": "Time After Modeling", "Sensor TL": "All", "start_timestamp": first_start_time,
    #                    "end_timestamp": modeled_time,
    #                    "elapsed_time": elapsed_modeling_time})
    # timestamps.append({"evento": "Time After Simulating", "Sensor TL": "All", "start_timestamp": first_start_time,
    #                    "end_timestamp": simulated_time,
    #                    "elapsed_time": elapsed_simulating_time})
    # df = pd.DataFrame(timestamps)
    # df.to_csv("time/tm_sim_time/timestamps-" + str(timeSlot.replace(':', '-')) + ".csv", index=False, sep=';')



#Function to convert geopoint format having a number without dots
def convert_format(value):
    return value.replace('.', '')  # Rimuovi solo il primo punto

# Function to convert a date expressed as a string in ISO format
import random
from datetime import datetime, timedelta

def convertDate(date: str, timeslot: str):
    """
    Convert a date and timeslot into an ISO 8601 formatted datetime string with a random delay.

    :param date: Date in 'yyyy-mm-dd' format.
    :param timeslot: Time slot in 'HH:MM-HH:MM' 24-hour format.
    :return: ISO 8601 formatted datetime string with a random delay.
    """
    # Parse the date in 'yyyy-mm-dd' format
    year, month, day = map(int, date.split("-"))
    # Extract start and end times in 24-hour format
    start_time, end_time = timeslot.split("-")
    end_hour_24 = int(end_time.split(":")[0])
    if end_hour_24 == 24:
        end_hour_24 = 0
        day += 1
    # Generate a random delay in minutes (between 0 and 10)
    random_delay = random.randint(0, 10)
    # Combine date and end time to create a datetime object for the end time of the timeslot
    base_time = datetime(year, month, day, end_hour_24, 0)
    # Add the random delay in minutes to the base time
    final_time = base_time + timedelta(minutes=random_delay)
    # Format as ISO 8601 with 24-hour time
    d = final_time.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    return d

# Funzione per attendere l'aggiornamento di modDate
def wait_for_mod_date_change(entry_id, old_mod_date, timeout=60, interval=1):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["orion-openiot"]
    collection = db["entities"]
    start_time = time.time()
    while time.time() - start_time < timeout:
        entry = collection.find_one({"_id.id": {"$regex": entry_id}})
        new_mod_date = entry["modDate"] if entry else None
        if new_mod_date != old_mod_date:
            return new_mod_date
        time.sleep(interval)  # Aspetta prima di ricontrollare
    return old_mod_date  # Timeout: ritorna il vecchio valore