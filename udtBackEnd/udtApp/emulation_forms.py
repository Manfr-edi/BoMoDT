import re
from django import forms
from mobilityvenv.emulation_data import load_data, select_data
from .forms import MACROMODEL_CHOICES, CAR_FOLLOWING_MODEL_CHOICES


class EmulationForm(forms.Form):
    start_date = forms.DateField(label='Dal giorno', widget=forms.DateInput(attrs={'type': 'date'}))
    end_date = forms.DateField(label='Al giorno (incluso)', widget=forms.DateInput(attrs={'type': 'date'}))
    pacing = forms.ChoiceField(label='Ritmo di trasmissione', choices=[
        ('day', 'Distribuisci una giornata nel tempo indicato'), ('interval', 'Intervallo tra invii / bulk')], initial='day')
    day_seconds = forms.FloatField(label='Secondi per giornata', initial=60, min_value=0, max_value=86400, required=False)
    interval_seconds = forms.FloatField(label='Secondi tra invii (0 = bulk)', initial=0, min_value=0, max_value=3600, required=False)
    start_hour = forms.IntegerField(label='Dalle ore', initial=0, min_value=0, max_value=23)
    end_hour = forms.IntegerField(label='Alle ore (escluse)', initial=24, min_value=1, max_value=24)
    sensor_ids = forms.CharField(label='Sensori (facoltativo)', required=False,
                                 help_text='Lascia vuoto per tutti, oppure TL2, TL645, …')
    skip_missing_days = forms.BooleanField(label='Salta i giorni senza dati', initial=True, required=False)
    continue_on_error = forms.BooleanField(label='Continua se un invio fallisce', initial=False, required=False)
    pipeline = forms.ChoiceField(label='Calcoli sulle misure trasmesse', initial='simulation', choices=[
        ('baseline', 'Solo baseline (invio → Grafana)'),
        ('macromodeling', 'Baseline + macromodeling'),
        ('simulation', 'Baseline + macromodeling + SUMO')])
    sumo_gui = forms.BooleanField(label='Apri SUMO con GUI', required=False, initial=False,
        help_text='Apre sumo-gui sul PC del backend per ogni scenario. Gli invii continuano in parallelo.')
    macromodel = forms.ChoiceField(label='Modello macroscopico', choices=MACROMODEL_CHOICES, initial='greenshield')
    car_following_model = forms.ChoiceField(label='Car-following', choices=CAR_FOLLOWING_MODEL_CHOICES, initial='Krauss')
    total_count = forms.IntegerField(label='Veicoli nello scenario SUMO', initial=10000, min_value=100, max_value=100000)
    tau = forms.FloatField(label='Tau', initial=1, min_value=0.1, max_value=10)
    sigma = forms.FloatField(label='Sigma (Krauss)', initial=0.5, min_value=0, max_value=1)
    sigma_step = forms.FloatField(label='Sigma step (Krauss)', initial=1, min_value=0.1)
    delta = forms.FloatField(label='Delta (IDM)', initial=4, min_value=1)
    stepping = forms.FloatField(label='Stepping (IDM)', initial=0.25, min_value=0.01)
    cc1 = forms.FloatField(label='CC1 (W99)', initial=1.3, min_value=0.1)
    cc2 = forms.FloatField(label='CC2 (W99)', initial=8, min_value=0.1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = load_data()
        self.available_dates = sorted(self.dataset['data'].unique())
        if self.available_dates:
            for name in ('start_date', 'end_date'):
                self.fields[name].initial = self.available_dates[0]
        for field in self.fields.values():
            if not isinstance(field.widget, forms.CheckboxInput):
                field.widget.attrs['class'] = 'form-control'

    def clean_sensor_ids(self):
        ids = list(dict.fromkeys(item.upper() for item in re.split(r'[,\s]+', self.cleaned_data.get('sensor_ids', '').strip()) if item))
        unknown = set(ids) - set(self.dataset['sensor_id'])
        if unknown:
            raise forms.ValidationError('Sensori non presenti: ' + ', '.join(sorted(unknown)))
        return ids

    def clean(self):
        data = super().clean()
        if self.errors:
            return data
        if data['end_date'] < data['start_date']:
            raise forms.ValidationError('Il giorno finale deve essere uguale o successivo a quello iniziale.')
        if (data['end_date'] - data['start_date']).days > 366:
            raise forms.ValidationError('Seleziona al massimo 367 giorni.')
        if data['start_hour'] >= data['end_hour']:
            raise forms.ValidationError('L’ora finale deve essere successiva a quella iniziale.')
        if data['pacing'] == 'day' and data['day_seconds'] is None:
            self.add_error('day_seconds', 'Indica i secondi per giornata.')
            return data
        if data['pacing'] == 'interval' and data['interval_seconds'] is None:
            self.add_error('interval_seconds', 'Indica l’intervallo (0 per bulk).')
            return data
        config = self.configuration(data)
        try:
            selected, missing = select_data(config, self.dataset)
        except ValueError as exc:
            raise forms.ValidationError(str(exc))
        self.selected_data = selected
        self.missing_days = missing
        return data

    @staticmethod
    def configuration(data):
        result = dict(data)
        result['start_date'] = data['start_date'].isoformat()
        result['end_date'] = data['end_date'].isoformat()
        return result

    def config(self):
        result = self.configuration(self.cleaned_data)
        result['total'] = len(self.selected_data) * (result['end_hour'] - result['start_hour'])
        result['days'] = sorted(self.selected_data['data'].unique())
        result['missing_days'] = self.missing_days
        result['sensor_count'] = int(self.selected_data['sensor_id'].nunique())
        return result
