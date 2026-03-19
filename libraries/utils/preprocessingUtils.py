import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.wkt import loads as load_wkt
from shapely.geometry import Point, shape, Polygon, mapping
from scipy.spatial import cKDTree
import numpy as np
import csv
import xml.etree.ElementTree as ET
import sys
import subprocess
from datetime import datetime
from libraries.constants import *
import sumolib
import os
import json
from xml.dom import minidom


def filterWithAccuracy(file_input: str, file_accuracy: str,date_column: str,sensor_id_column: str, output_file: str,  accepted_percentage: int):
    """
    Filter the traffic loop dataset using accuracy information.

    This function filters `file_input` (a traffic loop dataset) using `file_accuracy` (an accuracy dataset).
    Both files must contain columns for date and sensor ID to ensure proper filtering.
    Measurements with accuracy percentages below `accepted_percentage` are excluded.

    Args:
        file_input (str): Path to the input file (traffic loop dataset).
        file_accuracy (str): Path to the accuracy file.
        date_column (str): Name of the date column in both files.
        sensor_id_column (str): Name of the sensor ID column in both files.
        output_file (str): Path for the output filtered file.
        accepted_percentage (int): Minimum accepted percentage of accuracy for filtering.

    Returns:
        None: Saves the filtered data to `output_file`.
    """
    # Load the input data and accuracy data from the specified file paths
    df_input = pd.read_csv(file_input, sep=';', encoding="UTF-8")
    df_accuracy = pd.read_csv(file_accuracy, sep=';', encoding="UTF-8")

    # Clean and filter the accuracy data to only include measurements above `accepted_percentage`
    for ind, column in enumerate(df_accuracy.columns):
        if ind > 1:  # Ignore the first two columns, assumed to be `date_column` and `sensor_id_column`
            # Remove '%' and convert to integers
            df_accuracy[column] = df_accuracy[column].str.replace('%', '').astype(int)
            # Drop rows where the accuracy is below `accepted_percentage`
            df_accuracy = df_accuracy[df_accuracy[column] >= accepted_percentage]

    # Extract date and sensor ID columns as an identifier for filtering
    identifier = df_accuracy[[date_column, sensor_id_column]]
    keys = list(identifier.columns)

    # Create indices based on the date and sensor ID columns for both dataframes
    i1 = df_input.set_index(keys).index
    i2 = df_accuracy.set_index(keys).index

    # Filter `df_input` to include only rows that match the high-accuracy entries in `df_accuracy`
    filtered_df = df_input[i1.isin(i2)]

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created for the output file.")

    # Save the filtered data to the specified output file
    filtered_df.to_csv(output_file, sep=';', index=False)
    print(f"Output with filtered accuracy created at '{output_file}'.")

def generateRoadNamesFile(inputFile: str, sumoNetFile: str, roadNamesFilePath: str):
    """
    Generate a road names file with edge IDs linked to each road, based on geopoint coordinates.

    This function uses the coordinates provided in the input file to find the closest `edge_id` in the SUMO network.
    The `edge_id` is associated with each road name based on the provided coordinates. This information is saved in
    a CSV file.

    Args:
        inputFile (str): Path to the CSV file containing roads to map to an edge ID. This file should include
                         at least the columns 'Nome via' (road name) and 'geopoint' (latitude,longitude coordinates).
        sumoNetFile (str): Path to the SUMO network file (e.g., `net.xml`) that includes the road network.
        roadNamesFilePath (str): Path for the output CSV file where the mapped road names and edge IDs will be saved.

    Returns:
        None: The function saves two files:
              - A CSV file (`roadNamesFilePath`) with the road name and its corresponding edge ID.
    """
    # Load the SUMO network using sumolib
    net = sumolib.net.readNet(sumoNetFile)

    # Load input data and filter unique road names and geopoints
    input_df = pd.read_csv(inputFile, sep=';')
    df_unique = input_df[['Nome via', 'geopoint']].drop_duplicates()

    for index, row in df_unique.iterrows():
        # Extract latitude and longitude from the geopoint
        coord = row['geopoint']
        lat, lon = map(float, coord.split(','))

        # Convert geographic coordinates to SUMO's (x, y) coordinates
        x, y = net.convertLonLat2XY(lon, lat)
        print(f"SUMO reference coordinates (x, y): ({x}, {y})")

        # Find neighboring edges within a 25m radius, sorted by distance
        candidate_edges = net.getNeighboringEdges(x, y, r=25)
        edges_and_dist = sorted(candidate_edges, key=lambda edge: edge[1])

        # Attempt to find the closest edge that matches the road name or is suitable
        closest_edge = None
        if edges_and_dist:
            for edge, dist in edges_and_dist:
                edge_name = edge.getName().lower()
                road_name = row["Nome via"].lower()

                # Check if the edge name matches the road name, or find a suitable edge type
                if edge_name == road_name or edge.getType() not in ["highway.pedestrian", "highway.track",
                                                                    "highway.footway", "highway.path",
                                                                    "highway.cycleway", "highway.steps"]:
                    closest_edge = edge
                    break
            else:
                # If no matching edge is found, set closest_edge to the first suitable edge
                closest_edge = next((edge for edge, dist in edges_and_dist if
                                     edge.getType() not in ["highway.pedestrian", "highway.track", "highway.footway",
                                                            "highway.path", "highway.cycleway", "highway.steps"]), None)
        if closest_edge:
            print(f"Name: {closest_edge.getName()}")
            print(f"Edge ID: {closest_edge.getID()}")
            # Assign the closest edge ID to the row in df_unique
            df_unique.at[index, 'edge_id'] = closest_edge.getID()
        else:
            # Drop rows where no suitable edge is found within the network
            print(f"No suitable edge found for road '{row['Nome via']}' at coordinates ({lat}, {lon}).")
            df_unique.drop(index, inplace=True)

    # Save the updated DataFrame with edge IDs to the specified CSV file path
    df_unique.to_csv(roadNamesFilePath, sep=';', index=False)
    print(f"CSV file with road names and edge IDs saved at '{roadNamesFilePath}'")

def generateDetectorsCoordinatesFile(inputFile: str, detectorCoordinatesPath: str):
    """
    Generate a detector.csv file  that includes coordinates for each traffic loop.
    starting from the input file, all entries that have unique coordinates are extracted,
    allowing to distinguish all the loops in the measurement file.
    Args:
        inputFile: Path to the CSV file containing traffic loop to be stored. This file should include
                         at least the columns 'geopoint' (latitude, longitude coordinates).
        detectorCoordinatesPath: Path for the output csv file containing induction loop coordinates.
    Returns:
        None:
            the function saves a csv file (`detectorCoordinatesPath`) with induction loop definitions for SUMO.
    """

    # Load input data and filter unique road names and geopoints
    input_df = pd.read_csv(inputFile, sep=';')
    df_unique = input_df.drop_duplicates(['geopoint'])
    temp = []
    for index, row in df_unique.iterrows():
        id = row['ID_univoco_stazione_spira']
        # Extract latitude and longitude from the geopoint
        coord = row['geopoint']
        lat, lon = map(float, coord.split(','))

        temp.append({'id': id, 'lat': lat, 'lon': lon})

    new_df = pd.DataFrame(temp, columns=['id', 'lat', 'lon'])
    if not os.path.exists(MVENV_DATA_PATH):
        os.makedirs(MVENV_DATA_PATH)
        print(f"Directory '{MVENV_DATA_PATH}' created for the output file.")
    new_df.to_csv(detectorCoordinatesPath, sep=';', index=False)

def mapDetectorsFromCoordinates(sumoNetFile: str, detectorCoordinatesPath: str, detectorFilePath: str):
    """
    Generates the additional detector xml file.
    Through the script made available in SUMO, the coordinates of the traffic loops are mapped, enabling the creation of
    the detector.add.xml file useful for detecting traffic conditions within the simulator. Once the loops are mapped to
    the detectors, duplicate elements are removed (taking the unique values of the <lane, position> pair).
    Args:
        detectorCoordinatesPath: Path for the output csv file containing induction loop coordinates.
        detectorFilePath (str): Path for the output XML file containing induction loop definitions.

    Returns:
        None:
            the function saves an XML file (`detectorFilePath`) with induction loop definitions for SUMO.

    """
    print("starting to map the detector using geospatial coordinates...")
    script = SUMO_TOOLS_PATH + "/detector/mapDetectors.py"
    script = SUMO_TOOLS_PATH_UBUNTU + "/detector/mapDetectors.py"
    command = [
        sys.executable,
        script,
        "-n", sumoNetFile,
        "-d", detectorCoordinatesPath,
        "--det-output-file", SUMO_OUTPUT_PATH + "/detector.out.xml",
        "-o", detectorFilePath
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Script output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Script error:", e.stderr)

    # Dropping duplicates
    # Loading generated detector
    tree = ET.parse(detectorFilePath)
    root = tree.getroot()
    # Set for keeping seen detector
    seen = set()
    # loop for removing duplicates
    for element in list(root):
        # THIS IS SET FOR DETECTORS THAT EXCEED THE LENGTH OF THE ROAD ON WHICH THEY ARE ON.
        element.set('friendlyPos', 'True')
        key = (element.get('lane'), element.get('pos'))
        if key in seen:
            root.remove(element)
        else:
            seen.add(key)
    # save modified xml file
    tree.write(detectorFilePath, encoding='utf-8', xml_declaration=True)

def generateInductionLoopFile(inputFile: str, inductionLoopPath: str):
    """
    Generate a inductionLoop.csv file  A file that includes the information of traffic loops. Specifically, we extract
    the loops from the input file, detecting one loop for each unique <street name, loop id , coordinate> triplet.
    Args:
        inputFile: Path to the CSV file containing traffic loop to be stored. This file should include
                         at least the columns 'geopoint' (latitude, longitude coordinates).
        detectorCoordinatesPath: Path for the output csv file containing induction loop coordinates.
    Returns:
        None:
            the function saves a csv file (`detectorCoordinatesPath`) with induction loop definitions for SUMO.
    """

    # Load input data and filter unique road names and geopoints
    input_df = pd.read_csv(inputFile, sep=';')
    df_unique = input_df.drop_duplicates(['Nome via', 'ID_univoco_stazione_spira', 'geopoint'])
    temp = []
    for index, row in df_unique.iterrows():
        # Extract latitude and longitude from the geopoint
        coord = row['geopoint']
        lat, lon = map(float, coord.split(','))

        temp.append({'id': row['ID_univoco_stazione_spira'], 'roadname': row['Nome via'],'lat': lat, 'lon': lon})

    new_df = pd.DataFrame(temp, columns=['id', 'roadname', 'lat', 'lon'])
    new_df.to_csv(inductionLoopPath, sep=';', index=False)

def fillMissingEdgeId(roadnameFile: str):
    """
    Fill in missing edge IDs in the road names file.
    This function finds all entries in the specified road names file that lack an `edge_id`. For each entry with a missing
    `edge_id`, it attempts to find another entry in the same file with the same road name (`Nome via`) that has a valid
    `edge_id`. If such an entry is found, the missing `edge_id` is filled in with this value. The function also reports
    the number of entries that couldn't be assigned an `edge_id` because no matching road name with an `edge_id` was found.

    Args:
        roadnameFile (str): Path to the CSV file containing road names and edge IDs.

    Returns:
        None: The function updates the CSV file in place.

    :param roadnameFile: The path to the CSV file where road names and edge IDs are stored. The file is expected to
                         contain at least two columns:
                         - 'Nome via': The name of the road.
                         - 'edge_id': The unique identifier for each edge in the SUMO network.

    Side Effects:
        - Updates the input CSV file by filling in missing `edge_id` values where possible.
        - Prints the number of rows without an `edge_id` after attempting to fill them in.
    """
    # Load the road names data from the specified file
    df = pd.read_csv(roadnameFile, sep=';')
    empty = 0

    # Iterate over each row to check for missing edge IDs
    for index, row in df.iterrows():
        if pd.isnull(row['edge_id']):
            # Find other rows with the same road name ('Nome via') that have a valid edge ID
            sameStreet = df[(df['Nome via'] == row['Nome via']) & (df['edge_id'].notna())]
            if not sameStreet.empty:
                # Fill in the missing edge_id with the first matching edge_id found
                df.at[index, 'edge_id'] = sameStreet['edge_id'].values[0]
            else:
                # Increment count if no matching edge ID is found
                empty += 1

    # Report the number of roads without an edge ID
    print("Roads without edge ID: " + str(empty))
    # Save the updated DataFrame back to the CSV file
    df.to_csv(roadnameFile, sep=';', index=False)

def linkEdgeId(inputFile: str, roadnameFile: str, outputFile: str):
    """
    Link edge IDs from the road names file to each entry in the input file.

    This function matches entries from `inputFile` to entries in `roadnameFile` based on the road name (`Nome via`)
    and geographic point (`geopoint`). If a match is found, it assigns the corresponding `edge_id` to the entry in
    `inputFile`. Entries without a match are removed. The resulting data is saved to `outputFile`.

    Args:
        inputFile (str): Path to the input CSV file containing entries to which edge IDs will be linked.
                         This file should contain columns 'Nome via' and 'geopoint' for matching.
        roadnameFile (str): Path to the CSV file containing road names and edge IDs. This file should also
                            include 'Nome via', 'geopoint', and 'edge_id' columns.
        outputFile (str): Path for the output CSV file where the modified data with linked edge IDs will be saved.

    Returns:
        None: The function saves the updated data with edge IDs to `outputFile`.

    :param inputFile: The input CSV file containing data entries that need edge IDs.
    :param roadnameFile: The CSV file containing road names and associated edge IDs.
    :param outputFile: The path for the output CSV file where the updated data will be saved.

    Side Effects:
        - Saves the updated input data, with edge IDs linked where possible, to the specified `outputFile`.
    """
    # Load input and road names data
    df = pd.read_csv(inputFile, sep=';')
    dfRoadnames = pd.read_csv(roadnameFile, sep=';')

    # Iterate over each row in the input DataFrame to find and link edge IDs
    for index, row in df.iterrows():
        # Find matching edge_id in the road names file based on road name and geopoint
        edge = dfRoadnames.loc[
            (dfRoadnames['Nome via'] == row['Nome via']) &
            (dfRoadnames['geopoint'] == row['geopoint']),
            'edge_id'
        ]

        # If no matching edge_id is found, remove the row; otherwise, link the edge_id
        if len(edge) == 0:
            df.drop(index, inplace=True)
            continue
        df.at[index, 'edge_id'] = edge.values[0]  # Assign the first match

    # Save the modified DataFrame to the output file
    df.to_csv(outputFile, sep=';', index=False)
    print(f"Updated file with linked edge IDs saved at '{outputFile}'")

def filterForShadowManager(inputFile: str):
    """
    Filter and format data for the Shadow Manager.

    This function processes the input CSV file to extract and rename specific columns relevant to the Shadow Manager.
    It selects unique traffic loop entries based on the combination of road name (`RoadName`) and traffic loop ID (`TrafficLoopID`).
    The filtered and renamed data is then saved to a CSV file named `digital_shadow_types.csv` in the `SHADOW_TYPE_PATH` directory.

    Args:
        inputFile (str): Path to the CSV file containing raw traffic loop data with various columns.

    Returns:
        None: The function saves the processed data to `digital_shadow_types.csv` in the `SHADOW_TYPE_PATH` directory (resulting in the SHADOW_TYPE_FILE_PATH)

    :param inputFile: The path to the input CSV file.
    Side Effects:
        - Saves the processed and filtered data to a CSV file named `digital_shadow_types.csv` in the `SHADOW_TYPE_PATH` directory.
        - Creates the `SHADOW_TYPE_PATH` directory if it does not already exist.
    """
    # Load the input CSV file
    df = pd.read_csv(inputFile, sep=';')

    # Select and rename relevant columns for the Shadow Manager
    df = df[['Nodo da', 'Nodo a', 'Nome via', 'direzione', 'longitudine', 'latitudine', 'geopoint',
             'ID_univoco_stazione_spira', 'edge_id', 'codice_spira', 'Livello']]
    df.columns = ['StartingPoint', 'EndPoint', 'RoadName', 'Direction', 'Longitude', 'Latitude', 'Geopoint',
                  'TrafficLoopID', 'EdgeID', 'TrafficLoopCode', 'TrafficLoopLevel']

    # Remove duplicate entries based on the combination of RoadName and TrafficLoopID
    df = df.drop_duplicates(['RoadName', 'TrafficLoopID'])

    # Ensure the output directory exists
    if not os.path.isdir(SHADOW_TYPE_PATH):
        os.mkdir(SHADOW_TYPE_PATH)

    # Save the filtered DataFrame to a CSV file in the specified directory
    df.to_csv(os.path.join(SHADOW_TYPE_FILE_PATH), sep=';', index=False)

def generateRealFlow(inputFile: str):
    """
    Generate a real traffic flow file with selected columns.
    This function reads a CSV file containing traffic flow data, selects specific columns, and saves the filtered data
    to a new CSV file named `real_traffic_flow.csv` in the `REAL_TRAFFIC_FLOW_DATA_MVENV_PATH` directory.
    Args:
        :param inputFile: The path to the input CSV file containing traffic data.

    Returns:
        None: The function saves the filtered data to `real_traffic_flow.csv` in the specified directory.
    """
    # Load the input CSV file
    df = pd.read_csv(inputFile, sep=';')

    # Select the relevant columns for real traffic flow data
    columns_to_keep = [
        'data', 'codice_spira', '00:00-01:00', '01:00-02:00', '02:00-03:00', '03:00-04:00', '04:00-05:00',
        '05:00-06:00', '06:00-07:00', '07:00-08:00', '08:00-09:00', '09:00-10:00', '10:00-11:00', '11:00-12:00',
        '12:00-13:00', '13:00-14:00', '14:00-15:00', '15:00-16:00', '16:00-17:00', '17:00-18:00', '18:00-19:00',
        '19:00-20:00', '20:00-21:00', '21:00-22:00', '22:00-23:00', '23:00-24:00', 'Nome via', 'direzione',
        'longitudine', 'latitudine', 'geopoint', 'ID_univoco_stazione_spira', 'codZone'
    ]
    df = df[columns_to_keep]

    # Ensure the output directory exists
    os.makedirs(REAL_TRAFFIC_FLOW_DATA_MVENV_PATH, exist_ok=True)

    # Save the filtered DataFrame to the output CSV file with index label 'index'
    output_file = os.path.join(REAL_TRAFFIC_FLOW_DATA_MVENV_PATH, "real_traffic_flow.csv")
    df.to_csv(output_file, sep=';', index_label='index')
    print(f"Filtered real traffic flow data saved at '{output_file}'")

def generateEdgeDataFile(input_file: str, date: str = "01/02/2024", time_slot: str = "00:00-01:00", duration: str = '3600'):
    """
    Generate an XML `edgedata` file for the route sampler in Eclipse SUMO.
    This function creates an XML file that includes edge traffic data for the specified date and time slot, based on the
    vehicle count data in `input_file`. The output XML file is saved to `EDGE_DATA_FILE_PATH`.
    Args:
        :param input_file: Path to the CSV file containing traffic data. The file should contain:
                           - 'edge_id': ID of the road edge.
                           - 'data': Date of traffic measurement.
                           - Hourly time slot columns with vehicle counts (e.g., '00:00-01:00', '01:00-02:00').
        :param date: Date for which traffic data is extracted, formatted as 'dd/mm/yyyy'.
        :param time_slot: Time slot for traffic data extraction (e.g., '00:00-01:00').
        :param duration: Duration in seconds for the interval in the XML file. Default is '3600' (one hour).
    """
    # Create the root element for the XML structure
    root = ET.Element('data')
    interval = ET.SubElement(root, 'interval', begin='0', end=duration)

    # Load and filter data based on the specified date
    df = pd.read_csv(input_file, sep=';')
    df = df[df['data'].str.contains(date)]

    for index, row in df.iterrows():
        edge_id = str(row['edge_id'])
        first = int(time_slot[:2])
        last = int(time_slot[6:8])

        # Calculate the vehicle count for the specified time slot
        if last - first > 1:  # If the time slot spans multiple hours
            total_count = sum(row[f"{hour:02d}:00-{(hour + 1) % 24:02d}:00"] for hour in range(first, last))
            count = str(total_count)
        else:
            count = str(row[time_slot])

        # Add the edge element to the XML with the calculated count
        ET.SubElement(interval, 'edge', id=edge_id, entered=count)

    # Ensure the directory for the output file exists
    os.makedirs(os.path.dirname(EDGE_DATA_FILE_PATH), exist_ok=True)

    # Save the XML tree with indentation
    tree = ET.ElementTree(root)
    ET.indent(tree, '  ')
    tree.write(EDGE_DATA_FILE_PATH, encoding="UTF-8", xml_declaration=True)
    print(f"Edge data XML saved at '{EDGE_DATA_FILE_PATH}'")

def dailyFilter(inputFilePath: str, date: str):
    """
    Filter data by a specific date and save to a predefined daily traffic flow file.
    This function reads an input CSV file, filters rows based on the specified date, and saves the filtered data to `DAILY_TRAFFIC_FLOW_FILE_PATH'. The output file can be used
    within the Digital Twin environment for testing.
    Args:
        :param inputFilePath: Path to the input CSV file containing raw traffic data. Expected columns include:
        :param date: Date to filter the data by, formatted as 'dd/mm/yyyy'.
    """
    # Load the input CSV file
    df = pd.read_csv(inputFilePath, sep=';')

    # Filter data by the specified date
    df_filtered = df[df['data'].str.contains(date)]

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(DAILY_TRAFFIC_FLOW_FILE_PATH), exist_ok=True)

    # Save the filtered data to the daily traffic flow file path
    df_filtered.to_csv(DAILY_TRAFFIC_FLOW_FILE_PATH, sep=';', index=False)
    print(f"Filtered data for date '{date}' saved at '{DAILY_TRAFFIC_FLOW_FILE_PATH}'")

def reorderDataset(inputFilePath: str, outputFilePath: str):
    """
    Reorder the dataset in chronological order based on the 'data' column.
    Args:
        :param inputFilePath: Path to the input CSV file containing the dataset.
        :param outputFilePath: Path to the output file where the reordered dataset will be saved.
    """
    # Load the dataset
    df = pd.read_csv(inputFilePath, sep=';')

    # Ensure `data` column is in datetime format
    df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')

    # Sort the dataset by date
    df_sorted = df.sort_values(by='data')

    # Save the reordered dataset
    df_sorted.to_csv(outputFilePath, sep=';', index=False)
    print(f"Dataset reordered by date and saved at '{outputFilePath}'")

def filteringDataset(inputFilePath: str, start_date: str, end_date: str, outputFilePath: str):
    """
    Filter the dataset for a specific date range and save it to a specified file.
    Args:
        :param inputFilePath: Path to the input CSV file containing the dataset.
        :param start_date: Start date for filtering, formatted as 'mm/dd/yyyy'.
        :param end_date: End date for filtering, formatted as 'mm/dd/yyyy'.
        :param outputFilePath: Path where the filtered dataset will be saved.

    :side effect: Saves the filtered dataset to `outputFilePath`.
    """
    # Load the dataset
    df = pd.read_csv(inputFilePath, sep=';')

    # Convert `start_date` and `end_date` from 'mm/dd/yyyy' to 'yyyy-mm-dd' format for consistent comparison
    start_date = datetime.strptime(start_date, '%m/%d/%Y').strftime('%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%m/%d/%Y').strftime('%Y-%m-%d')

    # Ensure `data` column is parsed as datetime in 'dd/mm/yyyy' format
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')

    # Filter the dataset for the specified date range
    mask = (df['data'] >= start_date) & (df['data'] <= end_date)
    filtered_df = df.loc[mask]

    # Save the filtered dataset
    filtered_df.to_csv(outputFilePath, sep=';', index=False)
    print(f"Filtered data from {start_date} to {end_date} saved at '{outputFilePath}'")

def fillMissingDirections(inputFilePath: str, directionColumn = "direzione", defaultDirection = 'N'):
    """
    Fill missing direction in a traffic file. If a default direction is not set, North will be used.
    Args:
        inputFilePath: path to the input CSV file.
        directionColumn: column name where directions are defined.
        defaultDirection: direction to use when a road w.o. direction is met.

    Returns:
        The function updates the CSV file in place.
    """
    # Load the dataset
    df = pd.read_csv(inputFilePath, sep=';')
    # Replace empty values in the direction column with 'N'.
    df[directionColumn].fillna(defaultDirection, inplace=True)
    # Save modified dataset
    df.to_csv(inputFilePath, index=False, sep=';')


#Use 'Codice Area Statistica'as zoneColumnID for statistical areas
def addZones(inputFilePath: str, zoneFilePath: str, zoneColumn="codZone", zoneColumnID="COD_ZONA", withPlot=False):
    """
    Add Zone information to the input file entries. Each geopoint in the input data is evaluated as a point and searched
    for which geoshape (of the zone file) contains that point. Once found, the associated zone id information is added
    to the input file. For points that do not fall within any zone, the zone with the smallest distance centroid is
    associated.
    Args:
        inputFilePath: input file containing the geopoints to be associated
        zoneFilePath: file containing the zones described as geoshape and their associated IDs
        zoneColumn: column name to be created within the input file
        withPlot: boolean value for enable or not plot of the geopoint together with the geoshapes
    Returns: The function updates the CSV file in place.
    """
    print("Start adding zones for the input file...")
    df = pd.read_csv(inputFilePath, sep=';')
    # Converts the 'geopoint' column to geometry
    points_gdf = gpd.GeoDataFrame(
        df,
        geometry=df['geopoint'].apply(
            lambda x: Point(map(float, x.split(',')))
        ),
        crs="EPSG:4326"  # Assuming WGS84 geographical coordinates
    )
    # Inverts latitude with longitude (saved in reverse in the geopoint)
    points_gdf['geometry'] = points_gdf['geometry'].apply(lambda p: Point(p.y, p.x) if p.geom_type == 'Point' else p)
    points_gdf[zoneColumn] = {}

    # Custom function for creating geo shapes out of the zoneFile
    def parse_custom_geo_shape(geo_shape_str):
        """
        Converts a customised geo shape string into a Shapely geometry.

        Args:
            geo_shape_str (str): A string representing the geometry (e.g. JSON-like).

        Returns:
            shapely.geometry: A Shapely Geometric Object.
        """
        try:
            # Converts the string in a dict
            geo_data = eval(geo_shape_str)
            coordinates = geo_data["coordinates"]
            geom_type = geo_data["type"].lower()

            if geom_type == "polygon":
                return Polygon(coordinates[0])
            elif geom_type == "point":
                return Point(coordinates)
            else:
                raise ValueError(f"Unsupported geometry type: {geom_type}")
        except Exception as e:
            raise ValueError(f"Invalid geo shape format: {geo_shape_str}") from e

    geo_shapes_df = pd.read_csv(zoneFilePath, sep=';')
    geo_shapes_gdf = gpd.GeoDataFrame(
        geo_shapes_df,
        geometry=geo_shapes_df['Geo Shape'].apply(parse_custom_geo_shape),
        crs="EPSG:4326"  # Assuming WGS84 geographical coordinates
    )

    if withPlot:
        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot geoshapes as lines
        geo_shapes_gdf.boundary.plot(ax=ax, color='blue', linewidth=1)
        # Plot points
        points_gdf.plot(ax=ax, color='red', markersize=10)
        plt.show()

    # For each point, find the geoshape containing the point and add the zone ID
    for idx, point in points_gdf.iterrows():
        # Find the geoshape containing the point
        containing_shape = geo_shapes_gdf[geo_shapes_gdf.geometry.contains(point.geometry)]

        if not containing_shape.empty:
            # add the zone ID
            points_gdf.at[idx, zoneColumn] = containing_shape.iloc[0][zoneColumnID]  # sostituisci con il nome corretto della colonna
    print("Added found zone IDs. Filling the empty rows with the nearest Zone...")
    ### Filling NaN points
    # Extract geoshape centroids as an array of co-ordinates
    geo_shapes_centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in geo_shapes_gdf['geometry']])
    # Extract the coordinates of the points as an array
    points_coords = np.array([[geom.x, geom.y] for geom in points_gdf['geometry']])
    # Create a KDTree for geoshape centroids
    tree = cKDTree(geo_shapes_centroids)
    # Find the index of the nearest geoshape for each point
    distances, indices = tree.query(points_coords)
    # Match the ID of the geoshape closest to the points
    points_gdf['nearest_shape'] = indices
    points_gdf['nearest_shape'] = points_gdf['nearest_shape'].apply(lambda idx: geo_shapes_gdf.iloc[idx][zoneColumnID])
    # Replace NaN with the nearest geoshape
    points_gdf[zoneColumn] = points_gdf[zoneColumn].fillna(points_gdf['nearest_shape'])
    # Column no longer required
    points_gdf = points_gdf.drop(columns=['nearest_shape'])

    points_gdf.to_csv(inputFilePath, index=False, sep=';')
    print("Saved the new data.")


def fillEdgeDataInfo(inputFilePath: str, sumoNetFile: str):
    # Load the SUMO network using sumolib
    net = sumolib.net.readNet(sumoNetFile)

    # Parsing del file XML
    tree = ET.parse(inputFilePath)
    root = tree.getroot()
    # Itera attraverso tutti gli elementi <edge>
    for edge in root.findall(".//edge"):
        edge_id = edge.get("id")  # Leggi l'attributo 'id'
        qPKW = edge.get("qPKW")  # Leggi l'attributo 'qPKW'
        edge_lengths = {edge.getID(): edge.getLength() for edge in net.getEdges()}
        density = float(qPKW) / edge_lengths[edge_id]  # vehicles per km
        edge.set("density", str(density))
    # Salva l'output in un nuovo file XML
    output_file = "output_updated.xml"
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)


def generateFlow(inputFilePath: str, modelFilePath: str,outputFilePath: str, date: str, timeSlot: str):
    """
    Generate detector flow file starting from a traffic loop measurement file.
    Args:
        inputFilePath: path of the Traffic measurement file from which to take information such as detectorID and edgeID
        modelFilePath: path of the file that includes the information of the chosen traffic model
        outputFilePath: path of the file to save the generated flow data
        date: date of the selected traffic flow
        timeSlot: Time window value of the measurements to be evaluated reported in the format hh:mm-hh:mm
    Returns:
        None: generates the Flow file, saving it in the path indicated in outputFilePath
    """

    input_df = pd.read_csv(inputFilePath, sep=';')
    input_df = input_df[input_df['data'].str.contains(date)]
    df_model = pd.read_csv(modelFilePath, sep=';', decimal=',')
    data = []

    for index, row in input_df.iterrows():
        detectorID = row['ID_univoco_stazione_spira']
        edge_id = row['edge_id']
        gmodel = df_model.loc[df_model["edge_id"] == edge_id].iloc[0]
        vPKW = gmodel['velocity'] * 3.6
        first = int(timeSlot[:2])
        last = int(timeSlot[6:8])

        # Calculate the vehicle count for the specified time slot
        if last - first > 1:  # If the time slot spans multiple hours
            total_count = sum(row[f"{hour:02d}:00-{(hour + 1) % 24:02d}:00"] for hour in range(first, last))
            qPKW = str(total_count)
            time = (last - first) * 60
        else:
            qPKW = str(row[timeSlot])
            time = 60

        data.append({
            "Detector": detectorID,
            "Time": time,
            "qPKW": qPKW,
            "qLKW": 0,
            "vPKW": vPKW,
            "vLKW": 0
        })
    output_df = pd.DataFrame(data)
    output_df.to_csv(outputFilePath, sep=';', index=False, float_format='%.4f', decimal=',')

def generateEdgeFromFlow(inputFlowPath: str, detectorFilePath: str, outputEdgePath: str):
    script = r"C:\Program Files (x86)\Eclipse\Sumo\tools/detector/edgeDataFromFlow.py"
    detector = detectorFilePath
    flow = inputFlowPath
    output = outputEdgePath
    subprocess.run([sys.executable, script, "--detector-file", detector,
                    "--detector-flow-file", flow, "--output-file", output, "--flow-columns", "qPKW", "-i", '61'])


def exportZonesToSUMO(zoneFilePath: str, sumoNetFile: str, outputXmlPath: str,
                                    zoneColumnID="COD_ZONA", geoShapeColumn="Geo Shape"):
    """
    Converts zones from a CSV with GeoJSON-like polygon shapes to a SUMO-compatible .add.xml file using SUMO's own
    coordinate conversion.

    Args:
        zoneFilePath: Path to the CSV file with polygon shapes and zone IDs.
        sumoNetFile: Path to the SUMO .net.xml file.
        outputXmlPath: Path to save the output .add.xml file.
        zoneColumnID: Name of the column in the CSV with the zone ID.
        geoShapeColumn: Name of the column in the CSV with the GeoJSON-like polygon.
    """
    print("Reading SUMO network...")
    net = sumolib.net.readNet(sumoNetFile)

    print("Reading zones CSV...")
    df = pd.read_csv(zoneFilePath, sep=';')

    def parse_custom_geo_shape(geo_shape_str):
        geo_data = eval(geo_shape_str)  # GeoJSON-like dict
        coordinates = geo_data["coordinates"]
        if geo_data["type"].lower() == "polygon":
            return Polygon(coordinates[0])
        else:
            raise ValueError("Only 'Polygon' shapes are supported.")

    # Crea GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=df[geoShapeColumn].apply(parse_custom_geo_shape),
        crs="EPSG:4326"
    )

    # XML root
    root = ET.Element("additional")

    for _, row in gdf.iterrows():
        zone_id = str(row[zoneColumnID])
        polygon = row.geometry

        shape_coords = []
        for lon, lat in polygon.exterior.coords:
            x, y = net.convertLonLat2XY(lon, lat)
            shape_coords.append(f"{x:.2f},{y:.2f}")
        shape_str = " ".join(shape_coords)

        taz_elem = ET.SubElement(root, "taz", id=zone_id, shape=shape_str, color="51,128,255")
        ET.SubElement(taz_elem, "param", key="Id", value=zone_id)

    # Prettify XML
    rough_string = ET.tostring(root, encoding="unicode")
    parsed = minidom.parseString(rough_string)
    pretty_xml = parsed.toprettyxml(indent="  ")

    with open(outputXmlPath, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    print(f"TAZ file saved at: {outputXmlPath}")