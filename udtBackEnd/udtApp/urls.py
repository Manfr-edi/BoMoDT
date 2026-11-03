from django.urls import path

from . import views
from . import timing_views
from . import emulation_views
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    path('emulation/status', emulation_views.status, name='emulation_status'),
    path('emulation/start', emulation_views.start, name='emulation_start'),
    path('emulation/<str:session_id>/action', emulation_views.action, name='emulation_action'),
    path('emulation/<str:session_id>/export', emulation_views.export, name='emulation_export'),
    path('telemetry/rendered', timing_views.rendered, name='timing_rendered'),
    path('simulationResults/<str:folder_name>/timings', timing_views.execution_times, name='execution_times'),
    path("", views.index, name="index"),
    path("index", views.index, name="index"),
    path("monitor", views.monitor, name="monitor"),
    path("entityList", views.entityList, name='entityList'),
    path("entityList/<str:entity_id>", views.entity, name='entity'),
    path("simulationModeler", views.simulationModeler, name='simulationModeler'),
    path("simulation", views.simulation, name='simulation'),
    path("simulation/<str:folder>/", views.serve_image, name='serve_image'),
    path("simulationResults", views.simulationResults, name='simulationResults'),
    path("simulationResults/<str:folder_name>/", views.serveResults, name='serveResults'),
    path("image/<str:folder_name>/", views.serve_image, name="serve_image"),
    path('summary', views.mocksummary, name='summary'),
]
