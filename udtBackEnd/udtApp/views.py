from django.core.paginator import Paginator
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from django.core.paginator import Paginator
from django.http import FileResponse, Http404, JsonResponse
from django.http import HttpResponse
from django.shortcuts import render
from django.contrib import messages

from libraries.constants import PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH, SUMO_PATH, CONTAINER_ENV_FILE_PATH
from libraries.classes.DigitalTwinManager import DigitalTwinManager
from libraries.classes.DataManager import *
from libraries.classes.Planner import Planner
from libraries.classes.DigitalTwinManager import DigitalTwinManager
from libraries.classes.Agent import *
from libraries.classes.SumoSimulator import Simulator
from libraries.classes.SubscriptionManager import QuantumLeapManager
from libraries.classes.Broker import Broker
from libraries.utils.generalUtils import loadEnvVar
# from libraries.classes.DigitalTwinManager import configureCalibrateAndRun
from .forms import ConfigForm
from .models import Device
from libraries.utils.executionTiming import read_execution_times


def index(request):
    from .emulation_views import home
    return home(request)

def monitor(request):
    # TODO: modify the grafana dashboard
    return render(request, 'udtApp/monitor.html', {'nbar': 'monitor'})

def entity(request, entity_id):
    device = Device.objects(_id__id=entity_id).first()
    print(device)
    if device:
    # return HttpResponse("This is the page where the entity %s is shown" % entity_id)
        return render(request, 'udtApp/entity.html', {'device': device})
        # return HttpResponse(device)
    else:
        return HttpResponse("No device found with this ID", status=404)

def entityList(request):
    # Prendi il parametro 'type' dalla querystring
    device_type = request.GET.get('type', '')

    device_types = sorted(Device.objects.distinct('_id.type'))
    print(device_types)
    # Filtro per il tipo di dispositivo, se specificato
    if device_type:
        # deviceList = Device.objects.filter(_id__type__exact=device_type)
        deviceList = Device.objects.filter(__raw__={'_id.type': {'$regex': f".*{device_type}$"}})
    else:
        deviceList = Device.objects.all()

    print(f"Devices count: {deviceList.count()}")
    paginator = Paginator(deviceList, 10)
    pageNumber = request.GET.get('page')

    page_obj = paginator.get_page(pageNumber)

    context = {
        'page_obj': page_obj,
        'device_types': device_types,
        'request': request,  # Necessario per mantenere il filtro durante la paginazione
    }

    if deviceList:
        return render(request, 'udtApp/entityList.html', context)
    else:
        return HttpResponse("There is no device inside the DB")

def simulation(request):

    # Getting the absolute path of the scenarioCollection folder
    current_dir = os.path.abspath(os.getcwd())
    current_path = Path(current_dir).resolve()
    project_root = current_path.parent
    base_dir = project_root / 'sumoenv' / 'joined' / 'scenarioCollection'
    if not os.path.exists(base_dir):
        base_dir = project_root / 'MOBIDT' / 'sumoenv' / 'joined' / 'scenarioCollection'
        if not os.path.exists(base_dir):
            return render(request, 'udtApp/emptyPage.html', {'item': 'Scenario folder'})

    # Get the selected type (in the page filter). Default is 'basic'
    selected_type = request.GET.get('type', 'basic')
    selected_date = request.GET.get('date', '').strip()
    if not selected_date:  # Se la data non è fornita, usa la data corrente
        selected_date = datetime.now().strftime('%Y-%m-%d')
    start_time = request.GET.get('start_time', '00:00').strip()  # rimuovere spazi bianchi
    end_time = request.GET.get('end_time', '24:00').strip()

    today = datetime.now().date()
    default_date = today.strftime('%Y-%m-%d')

    selected_start_datetime = datetime.strptime(f"{selected_date} {start_time}", '%Y-%m-%d %H:%M')
    selected_end_datetime = datetime.strptime(f"{selected_date} {end_time}", '%Y-%m-%d %H:%M')

    # List of folders containing 'basic' or 'congestioned' in the name
    folders = []
    for folder_name in os.listdir(base_dir):
        try:
            # La cartella ha una data nel formato YYYY-MM-DD_HH-MM-SS_tipo
            folder_datetime_str, folder_type = folder_name.rsplit('_', 1)
            folder_datetime = datetime.strptime(folder_datetime_str, '%Y-%m-%d_%H-%M-%S')

            # Verifica se la cartella è del tipo selezionato e se rientra nell'intervallo orario
            if (selected_type in folder_type) and (selected_start_datetime <= folder_datetime <= selected_end_datetime):
                folders.append(folder_name)
        except ValueError:
            # Se la cartella non ha il formato atteso, ignorala
            continue

    context = {
        'folders': folders,
        'selected_type': selected_type,
        'selected_date': selected_date,
        'selected_start_time': start_time,
        'selected_end_time': end_time,
        'default_date': default_date,
        'hours':  [f"{hour:02d}:00" for hour in range(24)],
    }
    return render(request, 'udtApp/simulation.html', context)


# def serve_image(request, folder):
#
#     current_dir = os.path.abspath(os.getcwd())
#     current_path = Path(current_dir).resolve()
#     project_root = current_path.parent
#     base_dir = project_root / 'sumoenv' / 'joined' / 'scenarioCollection'
#     folder_path = os.path.join(base_dir, folder)
#
#
#     # Filter only .png files
#     images = [img for img in os.listdir(folder_path) if img.endswith(".png")]
#
#     # Get the selected type (in the page filter). Default is 'basic'
#     selected_type = request.GET.get('type', 'basic')
#
#     context = {
#         'folder': folder,
#         'images': images,
#         'selected_type': selected_type,
#         'base_dir': base_dir
#     }
#     return render(request, 'udtApp/simulationScenario.html', context)

def simulationModeler(request):
    envVar = loadEnvVar(CONTAINER_ENV_FILE_PATH)
    timescalePort = envVar.get("TIMESCALE_DB_PORT")
    if request.method == 'POST':
        timescaleManager = TimescaleManager(
            host="localhost",
            port=timescalePort,
            dbname="quantumleap",
            username="postgres",
            password="postgres"
        )
        dataManager = DataManager("TwinDataManager")
        dataManager.addDBManager(timescaleManager)
        configurationPath = SUMO_PATH + "/standalone"
        logFile = SUMO_PATH + "/standalone/command_log.txt"
        sumoSimulator = Simulator(configurationPath=configurationPath, logFile=logFile)
        twinManager = DigitalTwinManager(dataManager=dataManager, simulator=sumoSimulator,
                                         sumoConfigurationPath=configurationPath, sumoLogFile=logFile)
        form = ConfigForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            time_slot = [int(data['start_time']), int(data['end_time'])]

            # Creiamo un dizionario con i parametri
            params = {
                'macromodel': data['macromodel'],
                'car_following_model': data['car_following_model'],
                'total_count': data['total_count'],
                'tau': str(data['tau']),
                'time_slot': time_slot,
                'date': data['data'].strftime('%Y-%m-%d'),
            }

            # Aggiunta degli additional parameters in base al modello selezionato
            if data['car_following_model'] == 'Krauss':
                additional_param = {'sigma': str(data['sigma']),
                                    'sigmaStep': str(data['sigma_step'])}
                folderResult = twinManager.configureCalibrateAndRun(dataFilePath=PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH,
                                                  macroModelType=params['macromodel'],
                                                  carFollowingModel=params['car_following_model'], tau=params['tau'],
                                                  parameters=additional_param,
                                                  date=params['date'], timeslot=time_slot, totalCount=params['total_count'])
                print("Executed simulation " + str(folderResult))
            elif data['car_following_model'] == 'IDM':
                additional_param = {'delta': str(data['delta']),
                                    'stepping': str(data['stepping'])}
                folderResult = twinManager.configureCalibrateAndRun(dataFilePath=PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH,
                                                  macroModelType=params['macromodel'],
                                                  carFollowingModel=params['car_following_model'], tau=params['tau'],
                                                  parameters=additional_param,
                                                  date=params['date'], timeslot=time_slot, totalCount=params['total_count'])
                print("Executed simulation " + str(folderResult))
            elif data['car_following_model'] == 'W99':

                additional_param = {'cc1': str(data['cc1']),
                                    'cc2': str(data['cc2'])}
                folderResult = twinManager.configureCalibrateAndRun(dataFilePath=PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH,
                                                  macroModelType=params['macromodel'],
                                                  carFollowingModel=params['car_following_model'], tau=params['tau'],
                                                  parameters=additional_param,
                                                  date=params['date'], timeslot=time_slot, totalCount=params['total_count'])
                print("Executed simulation " + str(folderResult))
            params.update(additional_param)
            # Saving Model parameter data into the output folder
            params_file = os.path.join(folderResult, 'params.json')
            with open(params_file, 'w') as f:
                json.dump(params, f, indent=4)
            messages.success(request, "Simulation Completed!")
            folderResult = os.path.basename(os.path.normpath(folderResult))
            print(str(folderResult))

            # folderResultPath = os.path.join(SUMO_PATH, 'standalone', folderResult)
            # os.makedirs(folderResultPath, exist_ok=True)

            # return render(request, 'udtApp/result.html', {'result': context})
    else:
        form = ConfigForm()

    return render(request, 'udtApp/simulationModeler.html', {'form': form})
def serve_image(request, folder_name):
    BASE_DIR = "C:/Users/manfr/PycharmProjects/IntelliFlowTwin/sumoenv"
    """View per servire l'immagine 'plotResults.png' dalla cartella specificata."""
    image_path = os.path.join(BASE_DIR, folder_name, "plotResults.png")

    if os.path.exists(image_path):
        return FileResponse(open(image_path, "rb"), content_type="image/png")
    else:
        raise Http404("Immagine non trovata")
def simulationResults(request):
    from mobilityvenv.emulation_store import recent_sessions
    # Regex per il formato "yyyy_mm_dd_Nome1_Nome2"
    FOLDER_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_(.+)$")  # Nota il $ alla fine

    # Getting the absolute path of the scenarioCollection folder
    base_dir = Path(SUMO_PATH)
    base_dir.mkdir(parents=True, exist_ok=True)
    folders = []
    for folder in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, folder)):
            match = FOLDER_PATTERN.match(folder)
            if match:
                print(match)
                display_name = match.group(0).replace("_", " ").title()  # Rimuove underscore
                folders.append({"full_name": folder, "display_name": display_name})

    return render(request, "udtApp/simulationResults.html", {'folders': folders,
        'emulation_sessions': recent_sessions(), 'nbar': 'simulationResults'})



def serveResults(request, folder_name):
    FOLDER_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_(.+)$")  # Nota il $ alla fine
    # folder_path = os.path.join(settings.MEDIA_ROOT, "your_directory", folder_name)
    base_dir = Path(SUMO_PATH)
    folder_path = os.path.join(base_dir, folder_name)



    # Controlla che la cartella esista e rispetti il pattern
    if os.path.exists(folder_path):

        params_path = os.path.join(folder_path, 'params.json')
        # Check if param.json file exists
        if not os.path.exists(params_path):
            return JsonResponse({'error': 'params.json not found'}, status=404)
        # parameters reading
        with open(params_path, 'r') as f:
            calibration_params = json.load(f)

        stats_file = os.path.join(folder_path, "mean_tripinfo_metrics.json")
        try:
            with open(stats_file, 'r') as f:
                stats_result = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            stats_result = {}  # fallback se il file non esiste o è corrotto
        csv_file = os.path.join(folder_path, "mean_errors.csv")
        # Leggi il file con separatore ;
        df = pd.read_csv(csv_file, sep=';')

        # Seleziona solo le colonne di interesse (se esistono)
        columns_to_keep = ['speed_nrmse', 'density_nrmse', 'flow_rmse', 'flow_geh']
        filtered_df = df[columns_to_keep]
        # Converto in lista di dizionari per il template
        if not filtered_df.empty:
            evaluation_results = filtered_df.iloc[0].to_dict()
        else:
            evaluation_results = {}

        execution_times = read_execution_times(folder_path)
        timing_rows = []
        if execution_times:
            baseline = execution_times.get('baseline', {})
            phases = execution_times.get('phases', {})
            timing_rows = [
                {'label': 'Baseline', 'seconds': baseline.get('mean_seconds'),
                 'detail': f"Mean send-to-display latency · {baseline.get('count', 0)} rendered / {baseline.get('sent_count', 0)} sent samples"},
                {'label': 'Macromodeling', 'seconds': phases.get('macromodeling'),
                 'detail': 'Total macroscopic model calculation across selected hours'},
                {'label': 'Simulation', 'seconds': phases.get('simulation'),
                 'detail': 'Scenario reconstruction, routes, SUMO startup and execution'},
            ]
            maximum = max((row['seconds'] or 0 for row in timing_rows), default=1) or 1
            for row in timing_rows:
                row['width'] = round(100 * (row['seconds'] or 0) / maximum, 2)

        # URL della dashboard grafana (modifica con la tua se necessario)
        grafana_url = "http://localhost:3000/d/l8-g1tiHk/"

        context = {
            'calibration': calibration_params,
            'stats': stats_result,
            'results': evaluation_results,
            'execution_times': execution_times,
            'timing_rows': timing_rows,
            'folder_name': folder_name,
            'grafana_url': grafana_url,
            'nbar': 'summary',
        }
        return render(request, 'udtApp/summary.html', context)
            # return render(request, "view_folder.html", {"folder_name": folder_name, "files": files})
    else:
        return render(request, "error.html", {"message": "Folder not found or access not allowed."})

def mocksummary(request):
    # Mock dei parametri
    calibration_params = {
        'macromodel': 'Greenshields',
        'car_following_model': 'Krauss',
        'tau': 1.0,
        'sigma': 0.5,
        'sigma_step': 2,
        'time_slot': ['15:00', '16:00'],
        'date': '2024-02-01'
    }

    # Mock dei risultati quantitativi
    evaluation_results = {
        'speed_nrmse': 20.35,
        'density_nrmse': 3.71,
        'flow_nrmse': 0.5,
        'flow_geh': 3.20
    }

    # Path placeholder per l'immagine
    result_image_url = '/static/img/mock_plot1.png'  # assicurati di avere questa immagine nella cartella static

    # URL della dashboard grafana (modifica con la tua se necessario)
    grafana_url = "http://localhost:3000/d/abc123/grafana-dashboard?orgId=1"

    context = {
        'calibration': calibration_params,
        'results': evaluation_results,
        'result_image_url': result_image_url,
        'grafana_url': grafana_url,
        'nbar': 'summary',
    }

    return render(request, 'udtApp/summary.html', context)


def backup_serveResults(request, folder_name):
    FOLDER_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_(.+)$")  # Nota il $ alla fine
    # folder_path = os.path.join(settings.MEDIA_ROOT, "your_directory", folder_name)
    current_dir = os.path.abspath(os.getcwd())
    current_path = Path(current_dir).resolve()
    project_root = current_path.parent
    base_dir = project_root / 'sumoenv'
    folder_path = os.path.join(base_dir, folder_name)
    params_path = os.path.join(folder_path, 'params.json')

    # Controlla che il file esista
    if not os.path.exists(params_path):
        return JsonResponse({'error': 'params.json not found'}, status=404)

    # Lettura dei parametri
    with open(params_path, 'r') as f:
        params = json.load(f)

    # Controlla che la cartella esista e rispetti il pattern
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)

        # Find a .png file inside the directory
        image_file = None
        for file in files:
            if file.endswith(".png"):
                image_file = file
                break
        csv_file = os.path.join(folder_path, "mean_errors.csv")
        table_data = None
        headers = []
        if os.path.exists(csv_file):  # Se il file esiste, lo carica
            df = pd.read_csv(csv_file, sep=';', decimal=',')  # Legge il CSV
            table_data = df.to_dict(orient="records")  # Converte in lista di dizionari
            headers = df.columns.tolist()  # Prende le intestazioni
        context = {
            "folder_name": folder_name,
            "image_url": f"/media/{folder_name}/{image_file}" if image_file else None,
            "table_data": table_data,
            "headers": headers
            }
        print(f"{folder_path}/{image_file}")
        return render(request, "udtApp/result.html", context)
            # return render(request, "view_folder.html", {"folder_name": folder_name, "files": files})
    else:
        return render(request, "error.html", {"message": "Folder not found or access not allowed."})

