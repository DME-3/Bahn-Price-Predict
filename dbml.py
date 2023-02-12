import dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import datetime
import plotly.express as px
import os
import csv
import joblib

import numpy as np
import pandas as pd

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from datetime import date, timedelta

# additional dependency: scikit-learn==1.0.2

# Styling with dash_bootstrap_components
app = dash.Dash(external_stylesheets=[dbc.themes.SLATE], title='DB Price Predictions')

datetimeNow = datetime.datetime.now()

path = '/home/dme3/Bahn-Price-Predict/' if os.getenv('PYTHONANYWHERE_SITE') else './'
model_from_joblib = joblib.load(path + 'random_forest_prices_model_140123_1.pkl')

def get_holidays(date, time):
    weekday = None
    hour = None
    school_holiday_de = None
    public_holiday_de = None
    school_holiday_be = None
    public_holiday_be = None

    # Convert the date and time strings to datetime objects
    date = datetime.datetime.strptime(date, "%d.%m.%y")
    time = datetime.datetime.strptime(time, "%H:%M")

    # Get the weekday and hour from the datetime objects
    weekday = date.weekday() #strftime("%A")
    hour = time.hour

    # Read the holidays.csv file and look for the date
    with open(path + 'holidays-BE_DE-2022-2023.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            # Check if this row is for the specified date
            if row[0] == date.strftime("%d.%m.%y"):
                # Get the values for the school and public holidays
                school_holiday_de = row[1]
                public_holiday_de = row[2]
                school_holiday_be = row[3]
                public_holiday_be = row[4]
                break

    # Return the results
    return weekday, hour, school_holiday_de, public_holiday_de, school_holiday_be, public_holiday_be

predict_button = html.Div(
    [
        dbc.Button(
            "Predict!", id="predict-button", className="me-2", n_clicks=0
        ),
        html.Span(id="predict-output", style={"verticalAlign": "middle"}),
    ]
)

app.layout = html.Div([
    dbc.CardBody([
        html.H1("DB Price Prediction (Cologne - Brussels)", style={'textAlign': 'center'}),
        html.Br(),
        html.Br(),
        html.Div(children='''
            This app uses Machine Learning to predict ICE train prices (2nd class) between Brussels and Cologne. No guarantee is given as to the accuracy or reliability of those predictions!
        '''),
        html.Br(),
        html.Div(children='''
            The user must enter the travel date and time (the direction Cologne/Brussels or Brussels/Cologne does not matter). The price prediction chart will then be calculated when hitting the 'Predict!' button.
        '''),
        html.Br(),
        html.P('Model date: 140123'),
        html.Br(),
        dmc.Stack(
            children=[
                dmc.Text("Travel Date:", color="gray"),
                dmc.DatePicker(
                    id="date-picker",
                    minDate=datetimeNow.date() + timedelta(days=1),
                    maxDate=datetimeNow.date() + timedelta(days=60),
                    value=datetimeNow.date() + timedelta(days=21),
                    inputFormat="DD-MM-YYYY",
                    style={"width": 200},
                    persistence=True,
                ),
                dmc.Text("Departure Time:", color="gray"),
                dmc.TimeInput(style={"width": 100}, id = 'time-input', value=datetime.datetime.strptime("17:40", "%H:%M"), persistence=True, debounce=600),
            ],
        ),
        html.Br(),
        predict_button,
        html.Br(),
        dbc.Spinner(dcc.Graph(id="graph", config={'displayModeBar': False, 'scrollZoom': False})),
        dbc.Spinner(html.Div(id="output-test")),
        html.Br(),
        dbc.Row([
            dbc.Col(html.A('GitHub', href='https://github.com/DME-3/Bahn-Price-Predict', target='_blank'), width=3)
        ],
        style={'margin-left':'0px','margin-right':'0px'},
        justify='start')
    ]),
])

@app.callback(
    Output("graph", "figure"),
    Output("output-test", "children"),
    Input("predict-button", "n_clicks"),
    State('time-input', 'value'),
    State('date-picker', 'value'))
def update_predictions(clicks, user_time, user_date):

    user_date = datetime.datetime.strptime(user_date, '%Y-%m-%d')
    user_date_str = user_date.strftime('%d.%m.%y')

    user_time = datetime.datetime.strptime(user_time, '%Y-%m-%dT%H:%M:%S')
    user_time = user_time.strftime('%H:%M')

    weekday, hour, school_holiday_de, public_holiday_de, school_holiday_be, public_holiday_be = get_holidays(user_date_str, str(user_time))

    text = 'Weekday: ' + str(weekday) + ', hour: ' + str(hour) + ', school_holiday_de: ' + str(school_holiday_de) + ', school_holiday_be: ' + str(school_holiday_be) + ', public_holiday_de: ' + str(public_holiday_de) + ', public_holiday_be: ' + str(public_holiday_be)

    delta = user_date - datetimeNow

    max_dday = min(60, delta.days + 1)

    test_preds = []
    test_ddays = np.arange(max_dday,0,-1)
    test_dates = []

    for i in test_ddays:
        test_preds.append(model_from_joblib.predict(pd.DataFrame({'departure':[hour], 'weekday':[weekday], 'dday':[i], 'SchoolHolidayBE': [school_holiday_be], 'SchoolHolidayDE': [school_holiday_de], 'PublicHolidayBE': [public_holiday_be], 'PublicHolidayDE': [public_holiday_de]})).round(1)[0])
        test_date = user_date - timedelta(days = int(i))
        test_dates.append(test_date.strftime('%d.%m.%y'))

    arr = [test_dates, test_preds]

    df = pd.DataFrame(arr, index=['Ticket Buying Date', 'Predicted Price (€)'])
    df1 = df.T

    fig = px.line(df1, x='Ticket Buying Date', y='Predicted Price (€)')

    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x unified")
    fig.update(data=[{'hovertemplate' : 'Predicted Price: %{y:.2f} \u20ac<extra></extra>'}])

    return fig, text

if __name__ == '__main__':
    debug = False if os.getenv('PYTHONANYWHERE_SITE') else True
    app.run_server(debug=debug)