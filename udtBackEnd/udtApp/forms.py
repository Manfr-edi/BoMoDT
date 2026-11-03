from django import forms
import csv
from libraries.constants import PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH

# Definiamo le opzioni per gli enum
MACROMODEL_CHOICES = [
    ('greenshield', 'Greenshield'),
    ('underwood', 'Underwood'),
    ('vanaerde', 'Van Aerde'),
]

CAR_FOLLOWING_MODEL_CHOICES = [
    ('Krauss', 'Krauss'),
    ('IDM', 'IDM'),
    ('W99', 'Wiedemann'),
]

# Opzioni per gli slot orari (solo ore intere da 00 a 24)
TIME_SLOTS = [(i, f"{i}:00") for i in range(0, 25)]  # Da 00 a 24
class ConfigForm(forms.Form):
    macromodel = forms.ChoiceField(choices=MACROMODEL_CHOICES, label="Macromodel")
    car_following_model = forms.ChoiceField(choices=CAR_FOLLOWING_MODEL_CHOICES, label="Car Following Model")
    total_count = forms.IntegerField(
        label="Total Count",
        min_value=100, max_value=100000, initial=10000,
        widget=forms.NumberInput(attrs={'step': '100'})
    )
    tau = forms.FloatField(
        label="Tau",
        min_value=0.1, max_value=10, initial=1,
        widget=forms.NumberInput(attrs={'step': '0.1'})
    )

    # Additional parameters (saranno gestiti via JavaScript)
    sigma = forms.FloatField(
        label="Sigma (Krauss)", min_value=0, max_value=1, initial=0.5,
        widget=forms.NumberInput(attrs={'step': '0.1'}), required=False
    )
    sigma_step = forms.FloatField(
        label="Sigma Step (Krauss)", min_value=0.1, initial=1,
        widget=forms.NumberInput(attrs={'step': '0.1'}), required=False
    )

    delta = forms.FloatField(
        label="Delta (IDM)", min_value=1, initial=4,
        widget=forms.NumberInput(attrs={'step': '1'}), required=False
    )
    stepping = forms.FloatField(
        label="Stepping (IDM)", min_value=0.25, initial=0.25,
        widget=forms.NumberInput(attrs={'step': '0.01'}), required=False
    )

    cc1 = forms.FloatField(
        label="CC1 (W99)", min_value=0.1, initial=1.30,
        widget=forms.NumberInput(attrs={'step': '0.1'}), required=False
    )
    cc2 = forms.FloatField(
        label="CC2 (W99)", min_value=0.1, initial=8.0,
        widget=forms.NumberInput(attrs={'step': '0.1'}), required=False
    )

    data = forms.DateField(
        label="Date",
        widget=forms.DateInput(attrs={'type': 'date', 'format': 'yyyy-mm-dd'})
    )

    # Selezione dello slot orario
    start_time = forms.ChoiceField(choices=TIME_SLOTS, label="Starting Time")
    end_time = forms.ChoiceField(choices=TIME_SLOTS, label="Ending Time")

    def clean_data(self):
        selected_date = self.cleaned_data['data']
        with open(PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH, newline='', encoding='utf-8-sig') as source:
            available_dates = sorted({row['data'] for row in csv.DictReader(source, delimiter=';')})
        if selected_date.isoformat() not in available_dates:
            available_range = (
                f" Available dates range from {available_dates[0]} to {available_dates[-1]}; some days may be missing."
                if available_dates else " The dataset contains no measurements."
            )
            raise forms.ValidationError("No traffic measurements for the selected date." + available_range)
        return selected_date
    # Validazione personalizzata: start_time deve essere < end_time
    def clean(self):
        cleaned_data = super().clean()
        if cleaned_data.get("start_time") is None or cleaned_data.get("end_time") is None:
            return cleaned_data
        start = int(cleaned_data["start_time"])
        end = int(cleaned_data["end_time"])

        if start >= end:
            raise forms.ValidationError("The start time must be less than the end time.")

        return cleaned_data
