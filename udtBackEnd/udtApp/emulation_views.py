from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from uuid import UUID

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from mobilityvenv import emulation_store as store
from .emulation_forms import EmulationForm


def home(request):
    try:
        form = EmulationForm()
        dataset_error = None
    except (OSError, ValueError) as exc:
        form, dataset_error = None, str(exc)
    return render(request, 'udtApp/index.html', {'nbar': 'home', 'emulation_form': form,
        'dataset_error': dataset_error, 'emulation_sessions': store.recent_sessions()})


def session_payload(session):
    if session is None:
        return {'session': None}
    config = session['config']
    first = datetime.fromisoformat(config['start_date']).replace(tzinfo=timezone.utc)
    last = datetime.fromisoformat(config['end_date']).replace(tzinfo=timezone.utc) + timedelta(days=1)
    query = urlencode({'var-sessionID': session['id'], 'from': int(first.timestamp() * 1000),
                       'to': int(last.timestamp() * 1000), 'kiosk': '1'})
    return {'session': session, 'timings': store.summary(session['id']),
            'dashboard_url': 'http://localhost:3000/d/bomodt-emulation/emulation?' + query}


def status(request):
    session_id = request.GET.get('session_id')
    try:
        if session_id:
            session_id = str(UUID(session_id))
        result = session_payload(store.get_session(session_id))
    except ValueError:
        return JsonResponse({'error': 'Invalid session ID'}, status=400)
    response = JsonResponse(result)
    response['Cache-Control'] = 'no-store'
    return response


@require_POST
def start(request):
    try:
        form = EmulationForm(request.POST)
        if not form.is_valid():
            return JsonResponse({'errors': form.errors.get_json_data()}, status=400)
        session = store.create_session(form.config())
        return JsonResponse(session_payload(session), status=202)
    except (OSError, ValueError) as exc:
        return JsonResponse({'error': str(exc)}, status=409)


@require_POST
def action(request, session_id):
    try:
        session_id = str(UUID(session_id))
        store.control(session_id, request.POST.get('action'))
        return JsonResponse(session_payload(store.get_session(session_id)))
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=409)


def export(request, session_id):
    try:
        session = store.get_session(str(UUID(session_id)))
    except ValueError:
        session = None
    if session is None:
        return JsonResponse({'error': 'Session not found'}, status=404)
    response = JsonResponse(session_payload(session), json_dumps_params={'indent': 2})
    response['Content-Disposition'] = f'attachment; filename="emulation-{session["id"]}.json"'
    return response
