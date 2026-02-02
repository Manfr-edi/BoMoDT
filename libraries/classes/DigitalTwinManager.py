import sys
import time

import pandas as pd
from libraries.classes.SumoSimulator import Simulator
from libraries.classes.Planner import Planner
from libraries.classes.DataManager import DataManager
from typing import Optional
from libraries import constants
import os
import subprocess
from subprocess import Popen
from PIL import Image
import pytz
from datetime import datetime
from libraries.classes.TrafficModeler import TrafficModeler
from libraries.constants import SUMO_PATH, SUMO_NET_FILE_PATH, SUMO_ROUTES_PATH, projectPath, PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH, EDGE_DATA_FILE_PATH
from libraries.utils import statisticUtils
from libraries.utils.preprocessingUtils import generateEdgeDataFile

class DigitalTwinManager:
    """
    The DigitalTwinManager class orchestrates the interaction between historical traffic data, sumoenv simulations,
    and traffic planning to create a digital twin environment. This class is essential for running simulations
    based on real-world data to support traffic flow analysis.

    Attributes:
       sumoSimulator (Simulator): An instance of the Simulator class, which runs the sumoenv simulation.
       planner (Planner): An instance of the Planner class, responsible for scenario planning.
       dtDataManager (DataManager): An instance of the DataManager class, which handles data retrieval and storage.

    Methods:
       - __init__: Initializes the DigitalTwinManager by setting up the DataManager, Simulator, and Planner instances.
       - simulateBasicScenarioForOneHourSlot: Simulates a basic scenario for a one-hour time slot using historical data.
       - configureCalibrateAndRun: configure and calibrates macroscopic and car-following model. Then, a 24h simulation
       is executed and outputs are compared with the macroscopic values estimated before.
       - generateGraphs: function to generate graphs based on a specific simulated scenario.
       - showGraphs: function to show the generated graphs made up with generateGraphs.

   """

    sumoSimulator: Simulator
    planner: Planner
    dtDataManager: DataManager

    def __init__(self, dataManager: DataManager, simulator: Simulator, sumoConfigurationPath: str, sumoLogFile: str):
        """
        Initializes the DigitalTwinManager by creating instances of the DataManager, sumoenv simulator, and Planner.
        Args:
            :param dataManager: Optional name for the DataManager (default is "DataManager").
            :param sumoConfigurationPath: Path to the sumoenv configuration file.
            :param sumoLogFile: Path to the log file for sumoenv.
        """
        # TODO: manage the possibility to get as input directly the simulator and the planner instances.
        self.dtDataManager = dataManager
        configurationPath = SUMO_PATH + "/standalone"
        logFile = SUMO_PATH + "/sumo_log.txt"
        # self.sumoSimulator = Simulator(configurationPath=configurationPath, logFile=logFile)
        self.sumoSimulator = simulator
        self.planner = Planner(simulator=self.sumoSimulator)

    def simulateBasicScenarioForOneHourSlot(self, timeslot: str, date: str, entityType: str, totalVehicles: int,
                                            minLoops: int, congestioned: bool, activeGui: bool=False, timecolumn: Optional[str] = "timeslot"):
        """
        Simulates a basic traffic scenario for a one-hour time slot, retrieving historical data and using it to
        define the simulation parameters.
        Args:
            :param timeslot: The time slot for which historical traffic data is retrieved (e.g., "00:00-01:00").
            :param date: The date on which the traffic data is based (e.g., "2024-02-01").
            :param entityType: The type of entity being simulated (e.g., "roadsegment" or "device").
            :param totalVehicles: The total number of vehicles to include in the simulation.
            :param minLoops: The minimum number of simulation loops to perform.
            :param congestioned: A boolean flag to indicate if the simulation should include congestion.
            :param activeGui: Whether to activate the sumoenv graphical user interface (GUI) during the simulation.
            :param timecolumn: The name of the time column in the database used to retrieve historical data (default is "timeslot").

        Returns: A string representing the folder where the scenario is stored.
        """
        if entityType.lower() in ["road segment", "roadsegment"]:
            timescaleManager = self.dtDataManager.getDBManagerByType("TimescaleDBManager")
            df = timescaleManager.retrieveHistoricalDataForTimeslot(timeslot=timeslot, date=date, entityType=entityType, timecolumn=timecolumn)
            scenarioFolder = self.planner.planBasicScenarioForOneHourSlot(df, entityType=entityType, totalVehicles=totalVehicles, minLoops=minLoops, congestioned=False, activeGui=activeGui)
            return scenarioFolder

    def configureCalibrateAndRun(self, dataFilePath: str, carFollowingModel: str, macroModelType: str, tau: str,
                             parameters: {}, date: str, timeslot: [], totalCount = 10000):
        """
        Process of estimating traffic macroscopic values, calibrating traffic models and simulating them based on
        collected data. The function is designed to be applicable for an entire day's measurements or a smaller
        time window.
        It generates speed and density estimates using a selected :macroModelType. These data are then compared with the
        simulative outputs through the error evaluation function.
        Args:
            :param dataFilePath: input file containing measurements to be used for modeling and simulation activity
            :param carFollowingModel: the model type to calibrate and set for the simulation process
            :param macroModelType: the macromodel to apply to get flow, speed and density estimation
            :param tau: the headway time, in seconds, to be observed inside the custom car-follwing model
            :param parameters: additional parameters that are used to calibrate the car following model
            :param date: the date, in yyyy-mm-dd format, in which to go to evaluate the measurements
            :param timeslot: The time slot for which historical traffic data is retrieved (e.g., "00:00-01:00").

        Returns: returns the folder path in which simulations and results are stored.
        """
        basemodel = TrafficModeler(simulator=self.sumoSimulator, trafficDataFile=dataFilePath,
                                   sumoNetFile=SUMO_NET_FILE_PATH,
                                   date=date,
                                   timeSlot='00:00-01:00',
                                   modelType=macroModelType)
        print("Starting Configuration")
        # For each timeslot a TrafficModeler is set, hence constructing the macroscopic model
        for hour in range(timeslot[0], timeslot[1]):
            timeSlotFolder = ''
            if hour < 9:
                basemodel.changeTimeslot(timeSlot='0' + str(hour) + ':00-' + '0' + str(hour + 1) + ':00')
                timeSlotFolder = '0' + str(hour) + ':00-' + '0' + str(hour + 1) + ':00'
            elif hour == 9:
                basemodel.changeTimeslot('0' + str(hour) + ':00-' + str(hour + 1) + ':00')
                timeSlotFolder = '0' + str(hour) + ':00-' + str(hour + 1) + ':00'
            else:
                basemodel.changeTimeslot(str(hour) + ':00-' + str(hour + 1) + ':00')
                timeSlotFolder = str(hour) + ':00-' + str(hour + 1) + ':00'

            # Plot macroscopic data according to the selected macroscopic model fundamental diagram
            # basemodel.plotModel()

            # a car-following model is constructed, creating a specific file stored in the typeFilePath
            typeFilePath, confPath = basemodel.vTypeGeneration(modelType=carFollowingModel, tau=tau,
                                                           additionalParam=parameters)
            generateEdgeDataFile(input_file=PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH, date=date, time_slot=timeSlotFolder)
            self.planner.scenarioGenerator.generateRoute(inputEdgePath=EDGE_DATA_FILE_PATH,timeSlot=timeSlotFolder,
                                                         totalCount=totalCount)
            # the model values are saved in a .csv file
            basemodel.saveTrafficData(outputDataPath=typeFilePath + "/model.csv")
            timeSlotFolder = timeSlotFolder.replace(':', '-')
            first_param = str(list(parameters.values())[0])
            second_param = str(list(parameters.values())[1])
            folder_name = f"{date}_{macroModelType}_{carFollowingModel}_{tau}_{first_param}_{second_param}/{timeSlotFolder}"
            folder_path = os.path.join(SUMO_PATH, folder_name)
            output_path = folder_path + "/output/"
            os.makedirs(output_path, exist_ok=True)
            # Route folder construction
            timeslot_name = f"{timeSlotFolder}"
            routefolder_name = os.path.join(SUMO_PATH, 'routes')
            route_folder_path = os.path.join(routefolder_name, timeslot_name)
            print(route_folder_path)
            self.sumoSimulator.changeRouteFilePath(route_folder_path)
            self.sumoSimulator.start(activeGui=False, logFilePath=self.sumoSimulator.logFile)
            time.sleep(10)

        # confPath = projectPath + "/" + confPath
        paramvalues = list(parameters.values())
        # for each edge_id linked to a traffic loop, the simulation is evaluated according to the previous
        # macroscopic values. Simulation output of flow, speed and density are compared to the macroscopic ones
        df = pd.read_csv(typeFilePath + "/model.csv", sep=';', decimal=',')
        edge_ids = df[["edge_id"]].values
        for edge_id in edge_ids:
            os.makedirs(confPath + "/detected_output/", exist_ok=True)
            # Evaluate output according to macroscopic data. Data are saved in a dedicated folder
            basemodel.evaluateModel(edge_id=edge_id[0], confPath=confPath, outputFilePath=confPath + "/detected_output/" + str(edge_id[0]) + "_detectedFlow_t" + str(tau)
                                                                                   + "_ap" + str(paramvalues[0]) + "_ap" + str(paramvalues[1]) + ".csv")
            os.makedirs(confPath + "/error_output/", exist_ok=True)
            # the RMSE, MAPE of flow, speed and density are calculated. Additionally, squared R and GEH for flow is built
            basemodel.evaluateError(detectedFlowPath=confPath + "/detected_output/" + str(edge_id[0]) + "_detectedFlow_t" + str(tau) + "_ap" + str(paramvalues[0])
                                                   + "_ap" + str(paramvalues[1]) + ".csv",
                                    outputFilePath=confPath + "/error_output/" + str(edge_id[0]) + "_error_summary_t"+ str(tau)
                                                   + "_ap" + str(paramvalues[0]) + "_ap" + str(paramvalues[1]) + ".csv")
        # basemodel.plotTemporalResultsAverage(folderPath=confPath + "/detected_output", timeSlotRange=timeslot ,showImage=True)

        statisticUtils.evaluateSimulationError(confPath)
        statisticUtils.compute_mean_tripinfo_metrics(confPath)
        return confPath

    def generateGraphs(self, scenarioFolder: str):
        """
        Generate graphs based on the simulation outcome. The generated graphs show some info about the trajectory
        taken by the vehicles, the time spent in running/halted state and the depart delay time.
        Args:
            :param scenarioFolder: the folder where the simulated scenario output is stored
        Return:
        """
        graphScript = constants.SUMO_TOOLS_PATH + '/visualization/plotXMLAttributes.py'
        scenarioFolder = os.path.abspath(scenarioFolder)

        trajectory_cmd = [sys.executable, graphScript, "-x", "x", "-y", "y", "-o", scenarioFolder + "/traj_out.png",
                          scenarioFolder + "/fcd.xml", "--blind"]
        running_halted_cmd = [sys.executable, graphScript, scenarioFolder + "/summary.xml", "-x", "time", "-y",
                              "running,halting", "-o", scenarioFolder + "/plot_running.png", "--legend", "--blind"]
        depart_delay_cmd = [sys.executable, graphScript, "-i", "id", "-x", "depart", "-y", "departDelay",
                            "--scatterplot", "--xlabel", '"depart time [s]"', "--ylabel", '"depart delay [s]"',
                            "--ylim", "0,40", "--xticks", "0,1200,200,10", "--yticks", "0,40,5,10", "--xgrid",
                            "--ygrid", "--title", '"depart delay over depart time"', "--titlesize", "16",
                            scenarioFolder + "/tripinfos.xml", "--blind", "-o", scenarioFolder + "/departDelay.png"]

        commands = [trajectory_cmd, running_halted_cmd, depart_delay_cmd]
        procs = [Popen(i) for i in commands]
        for p in procs:
            p.wait()

    def showGraphs(self, scenarioFolder: str, saveSummary = False):
        """
        Groups the graphs previously generated and stored in the scenarioFolder and show them in one figure
        Args:
            :param scenarioFolder: the folder where the simulated scenario output is stored
        Returns:
        """
        scenarioFolder = os.path.abspath(scenarioFolder)

        # Graph list
        images = [scenarioFolder + '/traj_out.png', scenarioFolder + '/plot_running.png',scenarioFolder + '/departDelay.png']
        imgs = [Image.open(img) for img in images]

        # Group figures in one image
        widths, heights = zip(*(i.size for i in imgs))
        total_width = sum(widths)  # For horizontal concat
        max_height = max(heights)
        new_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in imgs:
            new_image.paste(img, (x_offset, 0))  # Incolla nella nuova immagine
            x_offset += img.width
        # Show Grouped image
        new_image.show()
        # Save new image
        if saveSummary:
            new_image.save(scenarioFolder + '/summary_image.png')
