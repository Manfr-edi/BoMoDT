"""Acknowledgements from the local Grafana timing panel (no session credentials)."""
import json
from urllib.parse import urlsplit
from uuid import UUID

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from libraries.utils.executionTiming import record_render
from libraries.utils.executionTiming import read_execution_times
from libraries.constants import SUMO_PATH
from pathlib import Path


@csrf_exempt
def rendered(request):
    origin = request.headers.get('Origin', '')
    # This deployment embeds local Grafana. Do not expose this endpoint cross-site.
    try:
        parsed = urlsplit(origin)
        allowed = parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1') and parsed.port == 3000
    except ValueError:
        allowed = False
    if not allowed:
        return JsonResponse({'error': 'Grafana origin required'}, status=403)
    if request.method == 'OPTIONS':
        response = JsonResponse({})
    elif request.method == 'POST':
        try:
            if len(request.body) > 65536:
                raise ValueError('Request too large')
            payload = json.loads(request.body)
            ids = payload['measurement_ids']
            if not isinstance(ids, list) or not 1 <= len(ids) <= 500:
                raise ValueError('Invalid measurement batch')
            ids = [str(UUID(item)) for item in ids]
            response = JsonResponse({'accepted': record_render(ids, payload['rendered_at_ms'])})
        except (KeyError, ValueError, TypeError, AttributeError):
            response = JsonResponse({'error': 'Invalid timing acknowledgement'}, status=400)
    else:
        response = JsonResponse({'error': 'POST required'}, status=405)
    response['Access-Control-Allow-Origin'] = origin
    response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response['Access-Control-Allow-Headers'] = 'Content-Type'
    response['Vary'] = 'Origin'
    return response


def execution_times(request, folder_name):
    root = Path(SUMO_PATH).resolve()
    folder = (root / folder_name).resolve()
    if folder.parent != root:
        return JsonResponse({'error': 'Invalid result folder'}, status=404)
    result = read_execution_times(folder)
    if result is None:
        return JsonResponse({'error': 'Execution times not recorded'}, status=404)
    response = JsonResponse(result)
    response['Cache-Control'] = 'no-store'
    return response
