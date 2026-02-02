import csv
import math
import os
import os.path
import numpy as np
import pandas as pd
import sumolib
from matplotlib import pyplot as plt
import sys
import subprocess
import xml.etree.ElementTree as ET
from scipy.interpolate import UnivariateSpline
from libraries.classes.SumoSimulator import Simulator
from libraries.constants import SUMO_PATH, SUMO_NET_FILE_PATH, SUMO_DETECTORS_ADD_FILE_PATH, SUMO_OUTPUT_PATH, SUMO_TOOLS_PATH
from pathlib import Path

class TrafficModeler:
    """
    A class that manages and compares traffic patterns based on the actual data provided as input. There are functions
    for constructing estimates from macroscopic models and functions for customizing car-following models used in the
    simulation phase

    Attributes:
        trafficData (pandas DataFrame): dataframe containing traffic measurement.
        macroscopicData (list): list of macroscopic data linked to specific induction loop location
        sumoNet (sumolib.net): sumolib class that contains the road network information representable on SUMO
        simulator (SumoSimulator): instance of SumoSimulator class that manages and runs SUMO simulations
        modelType (str): name of the macroscopic model type to apply when building estimations
        date (str): date on which the measurements to be modeled were taken
        timeslot (str): hourly timeslot on which the measurements to be modeled were taken
    """

    trafficData: pd.DataFrame
    macroscopicData: []
    sumoNet: sumolib.net
    simulator: Simulator
    modelType: str
    date: str
    timeSlot: str
    def __init__(self, simulator: Simulator, trafficDataFile: str, sumoNetFile : str, date: str = None, timeSlot: str = '00:00-23:00', modelType: str = "greenshield"):
        """
        Initializes the TrafficModeler, also deriving road parameters from the SUMO network

        :param trafficDataFile (str): path of the file containing traffic measurement.
        :param sumoNetFile (str): path of the file related to the sumo network (identified with the extension .net.xml).
        :param timeSlot (str): Time window value of the measurements to be evaluated reported in the format hh:mm-hh:mm
        :param modelType (str): Name of the traffic model to apply
        """
        self.simulator = simulator
        trafficDataDf = pd.read_csv(trafficDataFile, sep=';')
        if date is not None:
            self.trafficData = trafficDataDf[trafficDataDf['data'].str.contains(date)]
            self.date = date
        self.timeSlot = timeSlot
        self.timeSlot = self.timeSlot.replace(':', '-')
        self.sumoNet = sumolib.net.readNet(sumoNetFile)
        self.modelType = modelType
        self.getMacroscopicModel()


    def changeTimeslot(self, timeSlot: str):
        """
        change the timeslot to be evaluated. Changing the timeslot will cause a recalculation of macroscopic
        values according to the previously selected model
        """
        self.timeSlot = timeSlot
        self.timeSlot = self.timeSlot.replace(':', '-')
        self.getMacroscopicModel()

    def getMacroscopicModel(self):
        """
        calculate macroscopic data according to a selected model. This is called when a instance of the class is created
        or the timeslot is changed
        """
        self.macroscopicData = []
        for index, row in self.trafficData.iterrows():
            edge_id = row["edge_id"]
            edge = self.sumoNet.getEdge(edge_id)
            length = edge.getLength()
            vMax = edge.getSpeed() * 3.6
            laneCount = len(edge.getLanes())
            vehicleLength = 7.5  # 7.5 # this length is including the gap between vehicles
            maxDensity = (laneCount / vehicleLength) * 1000
            # print(f"Edge {edge_id}: k_jam = {maxDensity * 1000} vehicles/km")
            first = int(self.timeSlot[:2])
            last = int(self.timeSlot[6:8])
            # Calculate the vehicle count for the specified time slot
            if last - first > 1:  # If the time slot spans multiple hours
                total_count = sum(row[f"{hour:02d}:00-{(hour + 1) % 24:02d}:00"] for hour in range(first, last))
                flow = str(total_count)
            else:
                flow = str(row[self.timeSlot[:2]+':00-'+self.timeSlot[6:8]+':00'])

            vps = int(flow) / (3600 * (last - first))  # flow is set as vehicles per second
            density = int(flow) / vMax
            # density = int(flow) / ((length/1000) * laneCount)
            # density = density/1000
            laneDensity = density / laneCount
            laneVps = vps / laneCount
            if self.modelType == "greenshield":
                velocity = vMax * (1 - density / maxDensity)
            elif self.modelType == "underwood":
                velocity = vMax * np.exp(-density / (maxDensity/2))
            elif self.modelType == "vanaerde":
                critical_density = maxDensity / 2
                q_max = critical_density * vMax  # max flow
                c1 = (vMax / q_max) - (1 / maxDensity)
                c2 = 1 / (maxDensity * q_max)

                # Calcolo della velocità usando Van Aerde
                velocity = vMax / (1 + c1 * density + c2 * density ** 2)
            # velocity = velocity / 3.6
            #density = vps / velocity if velocity > 0 else maxDensity
            laneDensity = density / laneCount
            normVelocity = velocity / vMax
            vpsPerLane = vps / laneCount

            self.macroscopicData.append({
                "edge_id": edge_id,
                "length": length,
                "laneCount": laneCount,
                "flow": flow,
                "vehiclesPerSecond": vps,
                "vpsPerLane": vpsPerLane,
                "laneVps": laneVps,
                "density": density,
                "laneDensity": laneDensity,
                "maxDensity": maxDensity,
                "vMax": vMax,
                "velocity": velocity,
                "normVelocity": normVelocity
            })

    def saveTrafficData(self, outputDataPath: str):
    # TODO: set a name convention for saving new model data (e.g. greenshield_01-02-2024_00:00-23:00)
        """
        Save current traffic information stored inside the TrafficModeler into a .csv file
        Args:
            outputDataPath: path of the file to save the traffic model data
        """
        print("Saving...")
        df = pd.DataFrame(self.macroscopicData)
        df.to_csv(outputDataPath, sep=';', index=False, float_format='%.4f', decimal=',')
        print("New Model data saved into: " + outputDataPath + " file")


    def evaluateModel(self, edge_id: str, confPath: str, outputFilePath: str):
        """
        Determines the values found in simulations of speed, density, and flow of a specific traffic loop
        (given in input through an edge_id). The results, averaged for the one-hour slot, are saved in the given output
        .csv file.
        Args:
            :param edge_id: the edge_id on which to read measurements from simulations
            :param confPath: path in which the results of all hourly simulations are saved
            :param outputFilePath: path to the file in which to save simulation results for a loop
        Returns:
            no return, the function store the model data in a specific .csv file
        """
        # Remove a previous output file if present
        if os.path.isfile(outputFilePath):
            os.remove(outputFilePath)
        # Iterating through all one-hour simulation directories
        for folder_name in os.listdir(confPath):
            folder_path = os.path.join(confPath, folder_name)

            # Check if it is a valid directory
            if os.path.isdir(folder_path):
                if any(c.isalpha() for c in folder_name):
                    print(f"Cartella '{folder_name}' contiene lettere → salto")
                    continue
                xml_file = folder_path + "/output/edgedata-output.xml"
                print("PATH: " + str(xml_file))
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                except FileNotFoundError:
                    print(f"File {xml_file} not found → skip to the next folder")
                    continue
                except ET.ParseError:
                    print(f"Errore nel parsing del file {xml_file} → salto alla prossima cartella")
                    continue

                # edge_id = '151824728#0'
                # edge_id = '23288872#4' #the one used for experiments
                # edge_id = '-39673910' #this is the 30 km/h one but there are few measurements
                edgeData = []
                for interval in root.findall('interval'):
                    found = False
                    for edge in interval.findall('edge'):
                        if edge.get('id') == edge_id:  # Controlla se l'id corrisponde alla lane
                            found = True
                            detected_lane_density = edge.get('laneDensity')
                            detected_speed = float(edge.get('speed')) * 3.6
                            # detected_flow = int(edge.get('entered')) / 3600
                            # detected_density = detected_flow / float(detected_speed)
                            detected_density = float(edge.get('density'))
                            detected_flow = detected_speed * detected_density
                            detected_count = int(edge.get('entered'))
                            model_df = pd.read_csv(folder_path+"/model.csv", sep=';', decimal=',')
                            real_density = model_df[model_df['edge_id'] == edge_id]["density"].values[0]
                            real_speed = round(model_df[model_df['edge_id'] == edge_id]["velocity"].values[0], 2)
                            real_flow = model_df[model_df['edge_id'] == edge_id]["vehiclesPerSecond"].values[0]
                            real_count = model_df[model_df['edge_id'] == edge_id]["flow"].values[0]

                            edgeData.append({
                                'edge_id': edge_id,
                                'detected_density': detected_density,
                                'detected_lane_density': detected_lane_density,
                                'detected_speed': detected_speed,
                                'detected_flow': detected_flow,
                                'detected_count': detected_count,
                                'real_density': real_density,
                                'real_speed': real_speed,
                                'real_flow': real_flow,
                                'real_count': real_count,
                                'timeslot': folder_name
                            })
                    if not found:
                        edgeData.append({
                            'edge_id': edge_id,
                            'detected_density': '0',
                            'detected_lane_density': '0',
                            'detected_speed': '0',
                            'detected_flow': '0',
                            'detected_count': '0',
                            'real_density': '0',
                            'real_speed': '0',
                            'real_flow': '0',
                            'real_count': '0',
                            'timeslot': folder_name
                        })

                # Define the expected fieldnames in advance
                expected_fields = [
                    'edge_id', 'detected_density', 'detected_lane_density',
                    'detected_speed', 'detected_flow', 'detected_count',
                    'real_density', 'real_speed', 'real_flow', 'real_count', 'timeslot'
                ]
                if not edgeData:
                    print(f"Nessun dato valido per {edge_id} in {folder_path} → salto")
                    continue
                detector_df = pd.DataFrame(edgeData)
                if not all(key in edgeData[0] for key in expected_fields):
                    print(f"Incomplete data for {edge_id} in {folder_path} → jump to next")
                    continue
                else:
                    fieldnames = edgeData[0].keys()
                file_exists = os.path.isfile(outputFilePath)
                with open(outputFilePath, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=';')

                    # Write the header only if the file is new
                    if not file_exists:
                        writer.writeheader()
                    # Add data
                    writer.writerows(edgeData)
    def evaluateError(self, detectedFlowPath: str, outputFilePath: str):
        """
        Function for calculating RMSE and MAPE values of flow, velocity and density
        Args:
            :param detectedFlowPath: path of the file to get flow, speed and density attribute
            :param outputFilePath: path of the file to save error data
        Returns:
            no return, the functions stores in a specific .csv file the computed errors
        """
        error_data = []
        detector_df = pd.read_csv(detectedFlowPath, sep=';', decimal=',')
        # Initialise variables for RMSE and MAPE
        speed_squared_errors = 0
        speed_absolute_percentage_errors = 0
        density_squared_errors = 0
        density_absolute_percentage_errors = 0
        flow_squared_errors = 0
        flow_absolute_percentage_errors = 0
        n = 0  # Count valid data

        # Lists to store true values for calculating the range
        speed_true_values = []
        density_true_values = []

        # Iterate on traffic loops shared by model and SUMO data
        for id, row in detector_df.iterrows():
            speed_true = float(row["real_speed"])
            speed_pred = float(row["detected_speed"])
            density_true = float(row["real_density"])
            density_pred = float(row["detected_density"])
            flow_true = float(row["real_count"])
            flow_pred = float(row["detected_flow"])
            if speed_pred != 0 and speed_true != 0:
                speed_squared_errors += (speed_pred - speed_true) ** 2
                speed_absolute_percentage_errors += abs((speed_true - speed_pred) / speed_true)
                speed_true_values.append(speed_true)
            if density_pred != 0 and density_true !=0:
                density_squared_errors += (density_pred - density_true) ** 2
                density_absolute_percentage_errors += abs((density_true - density_pred) / density_true)
                density_true_values.append(density_true)
            if flow_pred !=0 and flow_true !=0:
                flow_squared_errors += (flow_pred - flow_true) ** 2
                flow_absolute_percentage_errors += abs((flow_true - flow_pred) / flow_true)
            if speed_pred == 0 or density_pred == 0 or flow_pred == 0:
                break
            n += 1
        # Get RMSE e MAPE
        if n > 0:
            speed_rmse = math.sqrt(speed_squared_errors / n)
            speed_mape = (speed_absolute_percentage_errors / n) * 100
            density_rmse = math.sqrt(density_squared_errors / n)
            density_mape = (density_absolute_percentage_errors / n) * 100
            flow_rmse = math.sqrt(flow_squared_errors / n)
            flow_mape = (flow_absolute_percentage_errors / n) * 100

            # Calculate NRMSE
            speed_range = 0
            density_range = 0
            if len(speed_true_values) != 0:
                speed_range = max(speed_true_values) - min(speed_true_values)
            if len(density_true_values) != 0:
                density_range = max(density_true_values) - min(density_true_values)

            speed_nrmse = speed_rmse / speed_range if speed_range != 0 else 0
            density_nrmse = density_rmse / density_range if density_range != 0 else 0

            print(f"Speed RMSE: {speed_rmse:.4f}")
            print(f"Speed MAPE: {speed_mape:.2f}%")
            print(f"Speed NRMSE: {speed_nrmse:.4f}")
            print(f"Density RMSE: {density_rmse:.4f}")
            print(f"Density MAPE: {density_mape:.2f}%")
            print(f"Density NRMSE: {density_nrmse:.4f}")
            print(f"Flow RMSE: {flow_rmse:.4f}")
            print(f"Flow MAPE: {flow_pred:.2f}%")
            error_data.append({'speed_rmse': speed_rmse,
                               'speed_mape': speed_mape,
                               'speed_nrmse': speed_nrmse,
                               'density_rmse':density_rmse,
                               'density_mape': density_mape,
                               'density_nrmse': density_nrmse,
                               'flow_rmse':flow_rmse,
                               'flow_mape':flow_mape})
            error_df = pd.DataFrame(error_data)
            error_df.to_csv(outputFilePath, sep=';', float_format='%.4f', decimal=',')

        else:
            print("No valid data for comparison.")
    def vTypeGeneration(self, modelType: str, tau: str = "1", additionalParam = {}):
        """
        Generates a vType with a specific car-following model. Depending on the model selected, two additional
        model-specific parameters can be set. The configured model is saved in a specific folder that shows the
        combination of models used and the date of simulation.
        Args:
            :param modelType: car-following model name (can be Krauss, IDM, or W99)
            :param tau: value of tau, that is driver's desired (minimum) time headway in seconds. Default value is 1
            :param additionalParam: list of additional parameters to configure the car-following model. There are two
            values and they vary from model to model
        Returns:
            it returns two path strings: a configuration path in which all simulation file are stored and a timeslot
            related path
        """
        first_param = str(list(additionalParam.values())[0])
        second_param = str(list(additionalParam.values())[1])
        folder_name = f"{self.date}_{self.modelType}_{modelType}_{tau}_{first_param}_{second_param}/{self.timeSlot}"
        conf_name = f"{self.date}_{self.modelType}_{modelType}_{tau}_{first_param}_{second_param}"
        # timeslot_folder = f"{self.timeSlot}"
        folder_path = os.path.join(SUMO_PATH, folder_name)
        conf_path = os.path.join(SUMO_PATH, conf_name)
        # folder_path = os.path.join(date_path, timeslot_folder)
        timeslot_name = f"{self.timeSlot}"
        route_path = os.path.join(SUMO_PATH + "/routes",timeslot_name)
        self.simulator.changeRouteFilePath(route_path)
        os.makedirs(folder_path, exist_ok=True)
        self.simulator.changeTypePath(folder_path)
        detector_path = os.path.join(SUMO_PATH, "static")
        self.simulator.changeDetectorPath(detector_path)
        # XML filename
        output_file = os.path.join(folder_path, "vtype.add.xml")
        root = ET.Element("additional")
        if modelType == "Krauss":
            self.carFollowingModelType = "Krauss"
            carFollowModel = "Krauss"
            vtypeID = "customModel"
            if "sigma" in additionalParam:
                print("FOUND SIGMA")
                sigma = additionalParam["sigma"]
            else:
                sigma = "0" #perfect driving
            if "sigmaStep" in additionalParam:
                print("FOUND SIGMA STEP")
                sigmaStep = additionalParam["sigmaStep"]
            else:
                sigmaStep = "step-length"
            vtype = ET.SubElement(root, "vType", {
                "id": vtypeID,
                "carFollowModel": carFollowModel,
                "tau": tau,
                "sigma": sigma,
                "sigmaStep": sigmaStep
            })
        elif modelType == "IDM":
            self.carFollowingModelType = "IDM"
            carFollowModel = "IDM"
            vtypeID = "customModel"
            vtype = ET.SubElement(root, "vType", {
                "id": vtypeID,
                "carFollowModel": carFollowModel,
                "tau": tau
            })
        elif modelType == "IDM":
            self.carFollowingModelType = "IDM"
            carFollowModel = "IDM"
            vtypeID = "customModel"
            if "delta" in additionalParam:
                delta = additionalParam["delta"]
            else:
                delta = "4"
            if "stepping" in additionalParam:
                stepping = additionalParam["stepping"]
            else:
                stepping = "0.25"
            vtype = ET.SubElement(root, "vType", {
                "id": vtypeID,
                "carFollowModel": carFollowModel,
                "tau": tau,
                "delta": delta,
                "stepping": stepping
            })
        elif modelType == "W99":
            self.carFollowingModelType = "W99"
            carFollowModel = "W99"
            vtypeID = "customModel"
            if "cc1" in additionalParam:
                 cc1 = additionalParam["cc1"]
            else:
                cc1 = "1.30"
            if "cc2" in additionalParam:
                cc2 = additionalParam["cc2"]
            else:
                cc2 = "8.0"
            vtype = ET.SubElement(root, "vType", {
                "id": vtypeID,
                "carFollowModel": carFollowModel,
                "tau": tau,
                "cc1": cc1,
                "cc2": cc2
            })
        # Writing to file
        tree = ET.ElementTree(root)
        ET.indent(tree, '  ')
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        print(f"vType File created: {output_file}")
        return folder_path, conf_path



### THESE ARE SOME PLOTTING FUNCTIONS.
    def plotModel(self, result: str = None):
        """
        function that plots the values of flux, density and velocity against each other in three graphs.
        The values are compared with the traffic model chosen at the initialization stage
        """
        print("Plotting the data according to theoretical model...")
        if result is None:
            df = pd.DataFrame(self.macroscopicData)
        else:
            df = pd.read_csv(result, sep=';', decimal=',')
        # unique values of max speed
        unique_vmax = df["vMax"].unique()

        # Create three figures for the three plot types
        # fig1, ax1 = plt.subplots(len(unique_vmax), 1, figsize=(8, 4 * len(unique_vmax)))
        fig2, ax2 = plt.subplots(len(unique_vmax), 1, figsize=(8, 4 * len(unique_vmax)))
        fig3, ax3 = plt.subplots(len(unique_vmax), 1, figsize=(8, 4 * len(unique_vmax)))

        if len(unique_vmax) == 1:  # Garantisci che gli assi siano array anche con un solo vmax
            # ax1 = [ax1]
            ax2 = [ax2]
            ax3 = [ax3]

        for i, v_max in enumerate(unique_vmax):
            # Filtra i dati per v_max corrente
            subset = df[df["vMax"] == v_max]
            v_max = (v_max * 3.6).round()
            # Calcola k_jam per ogni segmento basandosi sul numero di corsie
            # Media i valori di lane_count se ci sono più segmenti con lo stesso v_max
            avg_lane_count = subset["laneCount"].mean()
            # k_jam = 200 / avg_lane_count  # Stima densità al blocco
            # k_jam = 133 / 1000  # Densità massima (esempio: 133 veicoli/km)
            k_jam = avg_lane_count / 7.5  # Densità massima (esempio: 133 veicoli/km)
            # Dati di densità teorici (da 0 a k_jam)
            k = np.linspace(0, k_jam, 500)

            if self.modelType == "greenshield":
                v_theoretical = v_max * (1 - k / k_jam)
            elif self.modelType == "underwood":
                v_theoretical = v_max * np.exp(k / k_jam)
            elif self.modelType == "vanaerde":
                critical_density = k_jam / 2  # k_c = k_j / 2
                q_max = critical_density * v_max  # Flusso massimo
                c1 = (v_max / q_max) - (1 / k_jam)
                c2 = 1 / (k_jam * q_max)
                v_theoretical = v_max / (1 + c1 * k + c2 * k ** 2)

            q_theoretical = v_theoretical * k  # Flusso teorico


            # Flusso osservato

            if result is not None:
                q_observed = subset["detected_speed"] * 3.6 * subset["detected_density"]
            else:
                q_observed = subset["velocity"] * 3.6 * subset["density"]
            # Plot Velocità-Densità
            # ax1[i].plot(k, v_theoretical, label=f"Curva teorica v_max = {v_max} km/h", color='blue')
            # if result is not None:
            #     ax1[i].scatter(subset["detected_density"], (subset["detected_speed"] * 3.6), label="Dati osservati",
            #                    color='orange',
            #                    alpha=0.7)
            # else:
            #     ax1[i].scatter(subset["density"], (subset["velocity"] * 3.6), label="Dati osservati", color='orange',
            #                    alpha=0.7)
            # ax1[i].set_title(f"Velocità-Densità (v_max = {v_max} km/h)")
            # ax1[i].set_xlabel("Densità (veicoli/km)")
            # ax1[i].set_ylabel("Velocità (km/h)")
            # ax1[i].legend()
            # ax1[i].grid()

            # Plot Flusso-Densità
            ax2[i].plot(k, q_theoretical, label=f"Curva teorica v_max = {v_max} km/h", color='green')
            if result is not None:
                ax2[i].scatter(subset["detected_density"], q_observed, label="Dati osservati", color='red', alpha=0.7)
            else:
                ax2[i].scatter(subset["density"], q_observed, label="Dati osservati", color='red', alpha=0.7)
                # ax2[i].scatter(subset["density"], subset['vehiclesPerSecond'], label="Dati osservati", color='red', alpha=0.7)
            ax2[i].set_title(f"Flusso-Densità (v_max = {v_max} km/h)")
            ax2[i].set_xlabel("Densità (veicoli/km)")
            ax2[i].set_xlim(left=0, right=0.5)
            ax2[i].set_ylabel("Flusso (veicoli/h)")
            ax2[i].legend()
            ax2[i].grid()

            # Plot Flusso-Velocità
            ax3[i].plot(v_theoretical, q_theoretical, label=f"Curva teorica v_max = {v_max} km/h", color='purple')
            if result is not None:
                ax3[i].scatter((subset["detected_speed"] * 3.6), q_observed, label="Dati osservati", color='brown',
                               alpha=0.7)
            else:
                ax3[i].scatter((subset["velocity"] * 3.6), q_observed, label="Dati osservati", color='brown', alpha=0.7)
            ax3[i].set_title(f"Flusso-Velocità (v_max = {v_max} km/h)")
            ax3[i].set_xlabel("Velocità (km/h)")
            # ax3[i].set_xlim(left=0, right=70)
            ax3[i].set_ylabel("Flusso (veicoli/h)")
            ax3[i].legend()
            ax3[i].grid()

        # Migliora il layout delle figure
        # fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()

        # Mostra tutte le figure
        plt.show()


    def plotResults(self, resultFilePath: str):
        data = pd.read_csv(resultFilePath, delimiter=';')
        data = data.sort_values(by='detected_density')
        # Crea lo scatterplot
        plt.figure(figsize=(10, 6))
        # Interpolazione per i dati rilevati
        spl = UnivariateSpline(data['detected_density'], data['detected_flow'],
                               s=0)  # s=0 forza una perfetta interpolazione
        x_smooth = np.linspace(data['detected_density'].min(), data['detected_density'].max(), 500)
        y_smooth = spl(x_smooth)

        # Dati rilevati
        plt.scatter(data['detected_density'], data['detected_flow'], color='blue', label='Detected Flow', alpha=0.7)
        plt.plot(x_smooth, y_smooth, color='blue', linestyle='-', label='Detected Curve')
        # Dati reali
        plt.scatter(data['real_density'], data['real_flow'], color='red', label='Real Flow', alpha=0.7)

        # Etichette e legenda
        plt.title('Diagramma flusso-densità', fontsize=14)
        plt.xlabel('Density (vehicles/km)', fontsize=12)
        plt.ylabel('Flow (vehicles/hour)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # Mostra il grafico
        plt.tight_layout()
        plt.show()

    def plotTemporalResults(self, resultFilePath: str, showImage = True):
        def timeslot_to_numeric(timeslot):
            start_hour, start_min, end_hour, end_min = map(int, timeslot.split('-'))
            # Calcolo del tempo in secondi dall'inizio del giorno
            start_seconds = start_hour * 3600 + start_min * 60
            end_seconds = end_hour * 3600 + end_min * 60
            # Restituisci il valore medio del timeslot per l'ordinamento
            return (start_seconds + end_seconds) / 2

        data = pd.read_csv(resultFilePath, delimiter=';')

        data['timeslot_numeric'] = data['timeslot'].apply(timeslot_to_numeric)
        data = data.sort_values(by='timeslot_numeric')
        # Interpolazione per i dati rilevati

        spl1 = UnivariateSpline(data['timeslot_numeric'], data['detected_flow'], s=0)
        spl2 = UnivariateSpline(data['timeslot_numeric'], data['detected_speed'], s=0)
        spl3 = UnivariateSpline(data['timeslot_numeric'], data['detected_density'], s=0)
        x_smooth = np.linspace(data['timeslot_numeric'].min(), data['timeslot_numeric'].max(), 500)
        y_smooth1 = spl1(x_smooth)
        y_smooth2 = spl2(x_smooth)
        y_smooth3 = spl3(x_smooth)

        # Crea lo scatterplot
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig.legend(loc="upper left")
        plt.title(f"Modello {self.carFollowingModelType.capitalize()}: Speed and Flow over Time")
        # Primo asse: Velocità-Densità
        # ax1.plot(density_range, velocity_theoretical, label="Velocità teorica", color="blue")
        axes[0].scatter(data['timeslot_numeric'], data['detected_flow'], label="Detected Flow", color="blue", alpha=0.7)
        axes[0].plot(x_smooth, y_smooth1, color='blue', linestyle='-', label='Detected Curve')
        axes[0].scatter(data['timeslot_numeric'], data['real_flow'], label="Real Flow", color="red", alpha=0.7)

        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Flow (vehicles/h)")
        axes[0].tick_params(axis="y")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        # Secondo asse: Flusso-Densità
        # ax2 = ax1.twinx()
        axes[1].scatter(data['timeslot_numeric'], data['detected_speed'], label="Detected Speed", color="blue", alpha=0.7)
        axes[1].scatter(data['timeslot_numeric'], data['real_speed'], label="Real Speed", color="red", alpha=0.7)
        axes[1].plot(x_smooth, y_smooth2, color='blue', linestyle='-', label='Detected Curve')
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Speed (m/s)")
        axes[1].tick_params(axis="y", labelcolor="green")
        # axes[1].legend(loc="upper center")

        axes[2].scatter(data['timeslot_numeric'], data['detected_density'], label="Detected Density", color="blue",
                        alpha=0.7)
        axes[2].scatter(data['timeslot_numeric'], data['real_density'], label="Real Density", color="red", alpha=0.7)
        axes[2].plot(x_smooth, y_smooth3, color='blue', linestyle='-', label='Detected Curve')
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Density (vehicle/m)")
        axes[2].tick_params(axis="y", labelcolor="green")
        # axes[2].legend(loc="upper right")
        # Titolo e layout
        plt.tight_layout()
        simulationDirectory = os.path.dirname(resultFilePath)
        plt.savefig(simulationDirectory + '/plotResults.png')
        if showImage:
            plt.show()

        # # Dati rilevati
        # if y_axis == "flow":
        #     plt.scatter(data['timeslot_numeric'], data['detected_flow'], color='blue', label='Detected Flow', alpha=0.7)
        # elif y_axis == "speed":
        #     plt.scatter(data['timeslot_numeric'], data['detected_speed'], color='blue', label='Detected Speed', alpha=0.7)
        # elif y_axis == "density":
        #     plt.scatter(data['timeslot_numeric'], data['detected_density'], color='blue', label='Detected Density',
        #                 alpha=0.7)
        # plt.plot(x_smooth, y_smooth, color='blue', linestyle='-', label='Detected Curve')
        #
        # # Dati reali
        # if y_axis == "flow":
        #     plt.scatter(data['timeslot_numeric'], data['real_flow'], color='red', label='Real Flow', alpha=0.7)
        # elif y_axis == "speed":
        #     plt.scatter(data['timeslot_numeric'], data['real_speed'], color='red', label='Real Flow', alpha=0.7)
        # elif y_axis == "density":
        #     plt.scatter(data['timeslot_numeric'], data['real_density'], color='red', label='Real Density', alpha=0.7)
        #
        # # Etichette e legenda
        #
        # plt.xlabel('Time Slot (ordinato)', fontsize=12)
        # if y_axis == "flow":
        #     plt.title('Flow-Time Diagram with interpolation', fontsize=14)
        #     plt.ylabel('Flow (vehicles/hour)', fontsize=12)
        # elif y_axis == "speed":
        #     plt.title('Speed-Time Diagram with interpolation', fontsize=14)
        #     plt.ylabel('Speed (m/s)', fontsize=12)
        # elif y_axis == "speed":
        #     plt.title('Density-Time Diagram with interpolation', fontsize=14)
        #     plt.ylabel('Density (vehicles/km)', fontsize=12)
        # plt.legend()
        # plt.grid(True, linestyle='--', alpha=0.6)
        #
        # # Mostra il grafico
        # plt.tight_layout()
        # plt.show()

    def plotTemporalResultsAverage(self, folderPath: str, timeSlotRange: list[int], showImage=True):
        import os
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.interpolate import UnivariateSpline
        def timeslot_to_numeric(timeslot):
            start_hour, start_min, end_hour, end_min = map(int, timeslot.split('-'))
            start_seconds = start_hour * 3600 + start_min * 60
            end_seconds = end_hour * 3600 + end_min * 60
            return (start_seconds + end_seconds) / 2

        def is_timeslot_in_range(timeslot, start_hour, end_hour):
            start_h = int(timeslot.split('-')[0])
            return start_hour <= start_h < end_hour

        start_hour, end_hour = timeSlotRange
        data_frames = []
        if timeSlotRange[1] - timeSlotRange[0] < 3:
            print("It is not possible to plot temporal results having only one timeslot!")
            return
        for filename in os.listdir(folderPath):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(folderPath, filename), delimiter=';')
                df = df[df['timeslot'].apply(lambda ts: is_timeslot_in_range(ts, start_hour, end_hour))]
                if not df.empty:
                    df['timeslot_numeric'] = df['timeslot'].apply(timeslot_to_numeric)
                    data_frames.append(df)

        if not data_frames:
            print("No matching timeslots found in any CSV files.")
            return

        combined = pd.concat(data_frames)
        grouped = combined.groupby('timeslot_numeric').agg({
            'detected_flow': 'mean',
            'real_count': 'mean',
            'detected_speed': 'mean',
            'real_speed': 'mean',
            'detected_density': 'mean',
            'real_density': 'mean'
        }).reset_index()

        # Interpolazione
        spl1 = UnivariateSpline(grouped['timeslot_numeric'], grouped['detected_flow'], s=0)
        spl2 = UnivariateSpline(grouped['timeslot_numeric'], grouped['detected_speed'], s=0)
        spl3 = UnivariateSpline(grouped['timeslot_numeric'], grouped['detected_density'], s=0)

        x_smooth = np.linspace(grouped['timeslot_numeric'].min(), grouped['timeslot_numeric'].max(), 500)
        y_smooth1 = spl1(x_smooth)
        y_smooth2 = spl2(x_smooth)
        y_smooth3 = spl3(x_smooth)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig.legend(loc="upper left")
        plt.title(
            f"Modello {self.carFollowingModelType.capitalize()}: Speed and Flow from {start_hour}:00 to {end_hour}:00")

        axes[0].scatter(grouped['timeslot_numeric'], grouped['detected_flow'], label="Detected Flow", color="blue",
                        alpha=0.7)
        axes[0].plot(x_smooth, y_smooth1, color='blue', linestyle='-', label='Detected Curve')
        axes[0].scatter(grouped['timeslot_numeric'], grouped['real_count'], label="Real Flow", color="red", alpha=0.7)
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Flow (vehicles/h)")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle='--', alpha=0.6)

        axes[1].scatter(grouped['timeslot_numeric'], grouped['detected_speed'], label="Detected Speed", color="blue",
                        alpha=0.7)
        axes[1].scatter(grouped['timeslot_numeric'], grouped['real_speed'], label="Real Speed", color="red", alpha=0.7)
        axes[1].plot(x_smooth, y_smooth2, color='blue', linestyle='-', label='Detected Curve')
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Speed (km/h)")

        axes[2].scatter(grouped['timeslot_numeric'], grouped['detected_density'], label="Detected Density",
                        color="blue", alpha=0.7)
        axes[2].scatter(grouped['timeslot_numeric'], grouped['real_density'], label="Real Density", color="red",
                        alpha=0.7)
        axes[2].plot(x_smooth, y_smooth3, color='blue', linestyle='-', label='Detected Curve')
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Density (vehicle/m)")

        plt.tight_layout()
        plt.savefig(os.path.join(folderPath, 'plotResults.png'))
        if showImage:
            plt.show()
    def compareResults(self, resultPath: str, y_axis = "flow"):
        def timeslot_to_numeric(timeslot):
            start_hour, start_min, end_hour, end_min = map(int, timeslot.split('-'))
            # Calcolo del tempo in secondi dall'inizio del giorno
            start_seconds = start_hour * 3600 + start_min * 60
            end_seconds = end_hour * 3600 + end_min * 60
            # Restituisci il valore medio del timeslot per l'ordinamento
            return (start_seconds + end_seconds) / 2

        files = [f for f in os.listdir(resultPath) if f.startswith("detectedFlow") and f.endswith(".csv")]
        # Lista per salvare i DataFrame con i parametri estratti
        dfs = []
        detected_flow = []
        detected_speed = []
        detected_density = []
        spl_flow = []
        spl_speed = []
        spl_density = []
        y_smooth_flow = []
        y_smooth_speed = []
        y_smooth_density = []
        i = 0
        for file in files:
            # Extract file name without path
            filename = os.path.basename(file)

            # Extract the parameters from the file name
            parts = filename.replace(".csv", "").split("_")  # Divide il nome in base a "_"
            t_value = parts[1]  # "t1"
            ap_values = parts[2:]  # ["ap1", "ap5"]

            # Read the CSV
            data = pd.read_csv(resultPath + '/' + file, delimiter=';')
            data['timeslot_numeric'] = data['timeslot'].apply(timeslot_to_numeric)
            data = data.sort_values(by='timeslot_numeric')
            detected_flow.append(data['detected_flow'])
            detected_speed.append(data['detected_speed'])
            detected_density.append(data['detected_density'])
            spl_flow.append(UnivariateSpline(data['timeslot_numeric'], data['detected_flow'], s=0))
            spl_speed.append(UnivariateSpline(data['timeslot_numeric'], data['detected_speed'], s=0))
            spl_density.append(UnivariateSpline(data['timeslot_numeric'], data['detected_density'], s=0))
            x_smooth = np.linspace(data['timeslot_numeric'].min(), data['timeslot_numeric'].max(), 500)
            y_smooth_flow.append(spl_flow[i](x_smooth))
            y_smooth_speed.append(spl_speed[i](x_smooth))
            y_smooth_density.append(spl_density[i](x_smooth))
            i += 1


        # data = pd.read_csv(resultPath, delimiter=';')
        #
        # data['timeslot_numeric'] = data['timeslot'].apply(timeslot_to_numeric)
        # data = data.sort_values(by='timeslot_numeric')
        # # Interpolazione per i dati rilevati
        #
        # spl1 = UnivariateSpline(data['timeslot_numeric'], data['detected_flow_1'], s=0)
        # spl2 = UnivariateSpline(data['timeslot_numeric'], data['detected_speed_1'], s=0)
        # spl3 = UnivariateSpline(data['timeslot_numeric'], data['detected_density_1'], s=0)
        # spl4 = UnivariateSpline(data['timeslot_numeric'], data['detected_flow_2'], s=0)
        # spl5 = UnivariateSpline(data['timeslot_numeric'], data['detected_speed_2'], s=0)
        # spl6 = UnivariateSpline(data['timeslot_numeric'], data['detected_density_2'], s=0)
        # spl7 = UnivariateSpline(data['timeslot_numeric'], data['detected_flow_3'], s=0)
        # spl8 = UnivariateSpline(data['timeslot_numeric'], data['detected_speed_3'], s=0)
        # spl9 = UnivariateSpline(data['timeslot_numeric'], data['detected_density_3'], s=0)
        # x_smooth = np.linspace(data['timeslot_numeric'].min(), data['timeslot_numeric'].max(), 500)
        # y_smooth1 = spl1(x_smooth)
        # y_smooth2 = spl2(x_smooth)
        # y_smooth3 = spl3(x_smooth)
        # y_smooth4 = spl4(x_smooth)
        # y_smooth5 = spl5(x_smooth)
        # y_smooth6 = spl6(x_smooth)
        # y_smooth7 = spl7(x_smooth)
        # y_smooth8 = spl8(x_smooth)
        # y_smooth9 = spl9(x_smooth)
        # Crea lo scatterplot
        # fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        # fig.legend(loc="upper left")
        plt.figure(figsize=(14, 12))
        # Primo asse: Velocità-Densità
        # ax1.plot(density_range, velocity_theoretical, label="Velocità teorica", color="blue")
        # axes[0].scatter(data['timeslot_numeric'], detected_flow[0], label="Detected Standard Krauss Flow", color="blue", alpha=0.7)
        # axes[0].plot(x_smooth, y_smooth_flow[0], color='blue', linestyle='-', label='Detected Standard Krauss Curve')
        # axes[0].scatter(data['timeslot_numeric'], detected_flow[1], label="Detected Perfect Krauss Flow", color="red", alpha=0.7)
        # axes[0].plot(x_smooth, y_smooth_flow[1], color='green', linestyle='-', label='Detected Imperfect Krauss Curve')
        # axes[0].scatter(data['timeslot_numeric'], detected_flow[2], label="Detected Imperfect Krauss Flow", color="green", alpha=0.7)
        # axes[0].plot(x_smooth, y_smooth_flow[2], color='red', linestyle='-', label='Detected Perfect Krauss Curve')
        # axes[0].scatter(data['timeslot_numeric'], data['real_flow'], label="Real Flow", color="black", alpha=0.7)
        #
        # axes[0].set_xlabel("Time")
        # axes[0].set_ylabel("Flow (vehicles/h)")
        # axes[0].tick_params(axis="y")

        # axes[0].grid(True, linestyle='--', alpha=0.6)
        # Secondo asse: Flusso-Densità
        plt.scatter(data['timeslot_numeric'], detected_speed[0], label="Detected Krauss Speed", color="blue", alpha=0.7)
        plt.scatter(data['timeslot_numeric'], detected_speed[1], label="Detected Perfect Krauss Speed", color="red", alpha=0.7)
        plt.scatter(data['timeslot_numeric'], detected_speed[2], label="Detected Imperfect Krauss Speed", color="green", alpha=0.7)
        plt.scatter(data['timeslot_numeric'], data['real_speed'], label="Real Speed", color="black", alpha=0.7)
        plt.plot(x_smooth, y_smooth_speed[0], color='blue', linestyle='-', label='Detected Standard Krauss Curve')
        plt.plot(x_smooth, y_smooth_speed[1], color='red', linestyle='-', label='Detected Perfect Krauss Curve')
        plt.plot(x_smooth, y_smooth_speed[2], color='green', linestyle='-', label='Detected Imperfect Krauss Curve')
        plt.xlabel("Time")
        plt.ylabel("Speed (m/s)")
        plt.tick_params(axis="y", labelcolor="black")
        plt.legend(['Detected Standard Krauss Curve', 'Detected Perfect Krauss Curve', 'Detected Imperfect Krauss Curve', 'Real Speed'],loc="upper left")
        plt.title(f"Comparison Between Krauss and IDM model Speed (tau=1)")
        plt.show()

        plt.figure(figsize=(14, 12))  # Crea una nuova figura
        plt.scatter(data['timeslot_numeric'], detected_density[0], label="Detected Standard Krauss Density", color="blue",
                        alpha=0.7)
        plt.scatter(data['timeslot_numeric'], detected_density[1], label="Detected Perfect Krauss Density", color="red",
                        alpha=0.7)
        plt.scatter(data['timeslot_numeric'], detected_density[2], label="Detected Imperfect Krauss Density", color="green",
                        alpha=0.7)
        plt.scatter(data['timeslot_numeric'], data['real_density'], label="Real Density", color="black", alpha=0.7)
        plt.plot(x_smooth, y_smooth_density[0], color='blue', linestyle='-', label='Detected Standard Krauss Curve')
        plt.plot(x_smooth, y_smooth_density[1], color='red', linestyle='-', label='Detected Perfect Krauss Curve')
        plt.plot(x_smooth, y_smooth_density[2], color='green', linestyle='-', label='Detected Imperfect Krauss Curve')
        plt.xlabel("Time")
        plt.ylabel("Density (vehicle/m)")
        plt.tick_params(axis="y", labelcolor="black")
        plt.legend(['Detected Standard Krauss Curve', 'Detected Perfect Krauss Curve', 'Detected Imperfect Krauss Curve', 'Real Density'],loc="upper left")
        plt.title(f"Comparison Between Krauss and IDM model Density (tau=1)")
        # axes[2].legend(loc="upper right")
        # Titolo e layout

        plt.tight_layout()
        plt.show()


#Not sure if to leave it
    def evaluateModelwithDetector(self, detectorFilePath,  detectorOutputSUMO: str, outputFilePath: str):
            # Parse detector.add.xml
            tree = ET.parse(detectorFilePath)
            root = tree.getroot()

            # Estrarre il mapping detector -> edge
            detector_mapping = {}
            for detector in root.findall('inductionLoop'):
                detector_id = detector.get('id')
                lane = detector.get('lane')  # Es. "23276104_0"
                edge_id = lane.split('_')[0]  # get only edge_id
                detector_mapping[detector_id] = edge_id

            parser = ET.XMLParser(encoding='UTF-8')
            tree = ET.parse(detectorOutputSUMO, parser=parser)
            root = tree.getroot()
            detectorData = []
            for interval in root.findall("interval"):
                detector_id = interval.get('id')
                edge_id = detector_mapping[detector_id]
                flow = float(interval.get('flow'))
                speed = float(interval.get('speed'))
                vps = flow / 3600
                density = vps / speed

                if flow > 0:
                    detectorData.append({
                        'id': detector_id,
                        'edge_id': edge_id,
                        'flow': flow,
                        'meanSpeed': speed,
                        'density': density,
                        'occupancy': float(interval.get('occupancy'))
                    })

            detector_df = pd.DataFrame(detectorData)
            # detector_df.to_csv(outputFilePath, sep=';')
            model_df = pd.DataFrame(self.trafficData)

            # Confronta il modello con i detector per ID
            merged_df = pd.merge(model_df, detector_df, left_on='edge_id', right_on='edge_id', how='inner')

            # Calcola le discrepanze
            merged_df['flow_diff'] = merged_df['flow_x'].astype(int) - merged_df['flow_y'].astype(int)
            merged_df['velocity_diff'] = merged_df['velocity'] - merged_df['meanSpeed']

            merged_df = merged_df.rename(columns={"flow_x": "real_flow", "flow_y": "detected_flow", "velocity":
                "model_speed", "meanSpeed": "detected_speed", "density_x": "model_density", "density_y": "detected_density"})


            merged_df.to_csv(outputFilePath, sep=';', float_format='%.4f', decimal=',')
            print(merged_df[['edge_id', 'flow_diff', 'velocity_diff']])



