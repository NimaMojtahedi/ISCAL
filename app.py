# Frontend file

# required libraries
from datetime import time
from ntpath import join

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import pdb
import time
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
# internal files and functions
from utils import app_defaults, process_input_data, read_data_header
from models import Classifier

# dash library
import dash
from dash_bootstrap_components._components.InputGroup import InputGroup
from dash.exceptions import PreventUpdate
from dash import dcc
from dash import html
from dash import Input, Output, State, html, MATCH, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
# dash extension
from dash_extensions import Keyboard


import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots


# university logo path
University_Logo = "https://upload.wikimedia.org/wikipedia/de/9/97/Eberhard_Karls_Universit%C3%A4t_T%C3%BCbingen.svg"


# storage default params
params = app_defaults()

# create temp. save folder - this folder needs to be deleted after closing app
os.makedirs(os.path.join(
    params["current_directory"], "temp_saves"), exist_ok=True)
params["temp_save_path"] = os.path.join(
    params["current_directory"], "temp_saves")

# save params to the temp_saves folder
print("creating and storing app parameters.")
pd.DataFrame(params).to_json(os.path.join(
    params["temp_save_path"], "app_params.json"))


# define channels & dropdowns
def define_channels(channel_name=["No Channel in Data"], disabled=False, value=None):
    options = []
    dropdowns = []
    mins = []
    maxes = []
    if isinstance(channel_name[0], list):
        channel_name = channel_name[0]

    for nr, i in enumerate(channel_name):
        options.append({'label': i, 'value': i, 'disabled': disabled})
        dropdowns.append(dbc.Select(
            placeholder='N/A',
            options=[
                {"label": "Raw Data", "value": "raw"},
                {"label": "Filtered", "value": "flt"},
                {"label": "Both!", "value": "bth"},
            ],
            id={"type": "ddowns", "index": nr},
            disabled=True,
            style={"width": "110px", 'filter': 'blur(150px)', 'opacity': '0'},
            class_name='mb-2',
            size="sm",
        ))
        mins.append(dbc.Input(placeholder="Min", size="sm", id={"type": "mins", "index": nr}, disabled=True,
                              style={"width": "80px", 'filter': 'blur(150px)', 'opacity': '0'}, class_name='mb-2', inputmode='numeric', type="number",
                              min=0, max=1000))
        maxes.append(dbc.Input(placeholder="Max", size="sm", disabled=True,
                               style={"width": "80px", 'filter': 'blur(150px)', 'opacity': '0'}, class_name='mb-2', id={"type": "maxes", "index": nr},
                               inputmode='numeric', type="number", min=0, max=1000))

    components = dbc.Row(children=[
        dbc.Col(dbc.Checklist(
                id="channel_checklist",
                options=options,
                value=value,
                switch=True,
                inputStyle={"margin-right": "0px"},
                labelStyle={'display': 'block'},
                label_class_name='mb-3',
                style={'width': '110px', 'color': '#463d3b'}
                ),
                ),

        dbc.Col(dropdowns, class_name='me-4'),
        dbc.Col(mins, class_name='px-0 g-0 mx-0'),
        dbc.Col(maxes, class_name='px-0 g-0 mx-0'),
    ])
    return components


# navigation toolbar with logo, software title and a button
navbar = dbc.NavbarSimple(
    dbc.Container(
        dbc.Row(
            [

                dbc.Col(html.H1("ISCAL", style={'margin-left': '100px',
                        'color': '#003D7F', 'fontSize': 35})),


                dbc.Col(html.Div(
                    [
                        dbc.Button("Import",
                                   id="import-offcanvas-button", size="sm", n_clicks=0),
                        dbc.Offcanvas(children=[

                            html.P(
                                "1) Please customize the epoch length below:",
                                style={'color': '#463d3b'}
                            ),
                            dbc.InputGroup([dbc.InputGroupText("Epoch length"), dbc.Input(
                                placeholder="in seconds", autocomplete="off", id="epoch-length-input")],
                                class_name="mb-4"),

                            html.P(
                                "2) Please specify the sampling frequency:",
                                style={'color': '#463d3b'}
                            ),
                            dbc.InputGroup([dbc.InputGroupText("Sampling frequency"), dbc.Input(
                                placeholder="in Hz", autocomplete="off", id="sampling_fr_input")],
                                class_name="mb-4"),

                            html.P(
                                "3) Below are the channels of your dataset. "
                                "Please select which ones you want to load, "
                                "and then, press the Load button. "
                                "This can take a couple of minutes!",
                                style={'color': '#463d3b'}
                            ),

                            html.Div(children=define_channels(),
                                     id="channel_def_div"),

                            dbc.Row(children=[dbc.Button("Load", id="load_button", size="sm", n_clicks=0),
                                    html.Div(children=False, id="second_execution")],
                                    class_name="mt-3"),
                            html.Br(),
                            dcc.Loading(id="loading-state",
                                        children=html.Div(id="loading-output")),
                            dcc.Interval(
                                id='internal-trigger', interval=100, n_intervals=0, max_intervals=0),

                        ],
                            id="import-offcanvas",
                            title="Before you load, there are 3 steps...",
                            is_open=False,
                            backdrop='static',
                            scrollable=True,
                            style={'width': '500px',
                                   'title-color': '#463d3b', 'background': 'rgba(224, 236, 240, 0.2)', 'backdrop-filter': 'blur(10px)'}
                        ),
                    ]
                ), width="auto"),

                dbc.Col(dbc.Button("Export", id="save-button",
                        size="sm"), width="auto"),
                dbc.Col(html.Div(
                    [
                        dbc.Button("Advanced",
                                   id="advparam-button", size="sm", n_clicks=0),
                        dbc.Offcanvas(children=[
                            html.Div([
                                dbc.InputGroup([dbc.InputGroupText("ML algorithm"), dbc.Input(
                                    placeholder="Which ML you want to use?")], class_name="mb-1"),
                                dbc.InputGroup([dbc.InputGroupText("nThread"), dbc.Input(
                                    placeholder="The number of dedicated threads")], class_name="mb-1"),
                                dbc.InputGroup([dbc.InputGroupText("GPU"), dbc.Input(
                                    placeholder="1 for yes, 0 for no")], class_name="mb-1"),
                                dbc.InputGroup([dbc.InputGroupText("Lag time"), dbc.Input(
                                    placeholder="How much you can wait for training")], class_name="mb-1")
                            ]),
                            dbc.Row(dbc.Button("Apply", id="apply-params", size="sm"),
                                    class_name="mt-3"),

                        ],
                            id="advparam-offcanvas",
                            title="Here, you can customize the advance parameters!",
                            is_open=False,
                            backdrop='static',
                            scrollable=True,
                            placement='bottom',
                            style={
                                'title-color': '#463d3b', 'background': 'rgba(224, 236, 240, 0.2)', 'backdrop-filter': 'blur(10px)'}
                        ),
                    ]
                ), width="auto"),
                dbc.Col(html.A(dbc.Button("About Us", id="about-us-button", size="sm"), href="http://www.physiologie2.uni-tuebingen.de/", target="_blank"),
                        width="auto", style={'margin-left': '300px'}),
                dbc.Col(html.A(dbc.Button("Help", id="help-button", size="sm"),
                        href="https://github.com/NimaMojtahedi/Automatic-sleep-EEG-scoring", target="_blank"), width="auto"),
                dbc.Col(html.A(html.Img(src=University_Logo, height="40px"),
                        href="http://www.physiologie2.uni-tuebingen.de/", target="_blank"), width="auto"),
            ],
            align="center",
            justify="center",
        ),
        fluid=False,
    ),

    links_left=True,
    sticky="top",
    color="info",
    dark=False,
    fluid=False,
    className="mb-3",
)


inputbar = dbc.Nav(children=[
    dbc.Container(children=[
        dbc.Row(
            [
                dbc.Col(
                            [
                                dbc.Input(
                                    max=3,
                                    min=1,
                                    inputmode="numeric",
                                    type="number",
                                    id="minus-one_epoch",
                                    placeholder="",
                                    disabled=True,
                                    style={'width': '80px',
                                           'text-align': 'center'},
                                    size='sm'
                                ),
                            ],
                    class_name="d-flex justify-content-center",
                ),
                dbc.Col(
                    [
                        dbc.Input(
                            max=3,
                            min=1,
                            inputmode="numeric",
                            type="number",
                            id="null_epoch",
                            placeholder="",
                            disabled=True,
                            style={'width': '80px',
                                   'text-align': 'center'},
                            size='sm'
                        ),
                    ],
                    class_name="d-flex justify-content-center",
                ),

                dbc.Col(
                    [
                        dbc.Input(
                            max=3,
                            min=1,
                            inputmode="numeric",
                            type="number",
                            id="plus-one_epoch",
                            placeholder="",
                            disabled=True,
                            style={'width': '80px',
                                   'text-align': 'center'},
                            size='sm'
                        ),
                    ],
                    class_name="d-flex justify-content-center",
                ),
            ]),
        dbc.Row(
            dbc.Col(
                [
                    dbc.Input(
                        max=3,
                        min=1,
                        inputmode="numeric",
                        # type="number",
                        id="null_epoch_act",
                        placeholder="",
                        autocomplete="off",
                        style={'border': '2px solid', 'border-color': '#003D7F',
                               'width': '80px', 'text-align': 'center', 'hoverinfo': 'none'},
                        size='sm'

                    ),
                ],
                class_name="d-flex justify-content-center",
            ),
        ),
    ],
        fluid=True
    )],
    fill=True,
    pills=True,
    navbar=True,
    class_name="sticky-top",
)

graph_bar = dbc.Nav(dbc.Container(dbc.Row(
    dcc.Graph(id="ch", responsive=True, config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['zoomin', 'zoomout', 'zoom', 'pan'],
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        'scrollZoom': True,
        'editable': False,
        'showLink': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'EEG-Plot',
            'width': 3200,
            'scale': 1,
        }
    })
),
    class_name="fixed-bottom",
    style={"padding": "25px", "margin-bottom": "300px",
           "width": "100%", "height": "620px"},
    fluid=True))

# slidebar
sliderbar = dbc.Container(children=[
    dbc.Row(
        dcc.Slider(
            id='epoch-sliderbar',
            min=0,
            max=10,
            step=1,
            value=0,
            tooltip={"placement": "top", "always_visible": True}
        )
    )
],
    fluid=True,
)


def graph_channels(traces, names='Null Channel', downsamples=params["downsample"][0], s_fr=1000):
    # traces --> n * p array
    if downsamples:
        traces = traces[::downsamples, :]
        s_fr = int(s_fr / downsamples)

    # get trace length
    trace_len = len(traces[:, 0])

    # xaxis
    x_axis = np.linspace(0, trace_len/s_fr, trace_len)

    # number of channels
    nr_ch = traces.shape[1]

    # vertical ines positions
    x0 = (trace_len/3) / s_fr
    x1 = ((2 * trace_len) / 3) / s_fr

    y0 = []
    y1 = []
    y0 = [traces[:, i].min() for i in range(nr_ch)]
    y1 = [traces[:, i].max() for i in range(nr_ch)]

    fig = make_subplots(rows=nr_ch, cols=1, shared_xaxes=True,
                        print_grid=False, vertical_spacing=0.05)

    # changing px.line(y=trace)["data"][0] to go.Scatter(y=trace, mode="lines")
    # increase speed by factor of ~5

    for i in range(nr_ch):
        fig.add_trace(go.Scatter(x=x_axis, y=traces[:, i], mode="lines", line=dict(
            color='#003D7F', width=1), hoverinfo='skip'), row=i+1, col=1)

    for i in range(nr_ch):
        # adding lines (alternative is box)
        split_line1 = go.Scatter(x=[x0, x0], y=[y0[i], y1[i]], mode="lines",
                                 hoverinfo='skip',
                                 line=dict(color='black', width=3, dash='6px,3px,6px,3px'))
        split_line2 = go.Scatter(x=[x1, x1], y=[y0[i], y1[i]], mode="lines",
                                 hoverinfo='skip',
                                 line=dict(color='black', width=3, dash='6px,3px,6px,3px'))

        fig.add_trace(split_line1, row=i+1, col=1)
        fig.add_trace(split_line2, row=i+1, col=1)

        fig.update_layout(margin=dict(l=0, r=0, t=1, b=1),
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          showlegend=False,
                          xaxis_fixedrange=True,
                          )
        fig['layout'].update({'yaxis{}'.format(i+1): dict(title=names[i])})
    return fig


# get accuracy plot
def graph_ai_metric(data):
    # start plotting
    fig = px.line(y=data)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=0, r=0, t=0, b=0),
                      font={'size': 10, 'color': '#003D7F'},
                      xaxis={"automargin": True, "title_standoff": 0, "gridcolor": 'rgba(0,61,127,0.2)', "linewidth": 2, "linecolor": '#003D7F', "tickfont": {
                          'size': 12, 'color': '#003D7F'}, "title": "Iterations", "showgrid": True, "showline": True},
                      yaxis={"automargin": True, "title_standoff": 0, "gridcolor": 'rgba(0,61,127,0.2)', "linewidth": 2, "linecolor": '#003D7F', "tickfont": {
                          'size': 12, 'color': '#003D7F'}, "title": "Accuracy (%)", "showgrid": True, "showline": True},
                      xaxis_fixedrange=True,
                      yaxis_fixedrange=True,
                      showlegend=False,
                      )
    return fig


# spectrum & histograms
def graph_hs_ps(data, names='Null'):
    # list of 2 (power spectrums(by number of channels) and histograms(same))
    # first powerspectrum and then histogram

    spectrums = data[0]  # n*p
    histos = data[1]  # n*p
    nr_ch = histos.shape[1]

    # check if it is True
    assert spectrums.shape[1] == histos.shape[1]

    fig = make_subplots(rows=1, cols=nr_ch*2,
                        print_grid=False, vertical_spacing=0.0, horizontal_spacing=0.03,
                        # subplot_titles=("Power Spectrums", "", "", "Amplitude Histograms", "", "")
                        )
    # my_layout=dict(xaxis={"automargin":True,"title_standoff":0, "gridcolor":'rgba(0,61,127,0.2)',"linewidth":2, "linecolor": '#003D7F',"tickfont": {'size':12, 'color': '#003D7F'}, "title": "Iterations","showgrid": True, "showline": True})
    for i in range(nr_ch):
        # at the moment x is fixed to 30 Hz 120 = 30 * 4 (always 4)
        fig.add_trace(go.Scatter(x=np.linspace(0, 30, 120), y=spectrums[:, i], mode="lines", line=dict(
            color='black'), hoverinfo='skip'), row=1, col=i+1)
        fig.add_trace(go.Histogram(x=histos[:, i], marker_color='LightSkyBlue',
                                   opacity=0.75, histnorm="probability", hoverinfo='skip'), row=1, col=nr_ch+i+1)

    # in case it is necessary for histograms xbins=dict(start=-3.0,end=4,size=0.5)
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', bargap=0.25,
                      font={'size': 10, 'color': '#003D7F'},
                      xaxis={"automargin": True, "title_standoff": 0, "gridcolor": 'rgba(0,61,127,0.2)', "linewidth": 2, "linecolor": '#003D7F', "tickfont": {
                          'size': 12, 'color': '#003D7F'}, "title": "Frequency (Hz)", "showgrid": True, "showline": True},
                      yaxis={"automargin": True, "title_standoff": 0, "gridcolor": 'rgba(0,61,127,0.2)', "linewidth": 2, "linecolor": '#003D7F', "tickfont": {
                          'size': 12, 'color': '#003D7F'}, "title": "Spectral density", "showgrid": True, "showline": True},
                      showlegend=False,
                      xaxis_fixedrange=True,
                      yaxis_fixedrange=True,
                      )

    fig.update_yaxes(fixedrange=True, gridcolor='rgba(0,61,127,0.2)', linewidth=2, linecolor='#003D7F', tickfont={
                     'size': 12, 'color': '#003D7F'}, showgrid=True, showline=True)
    fig.update_xaxes(fixedrange=True, gridcolor='rgba(0,61,127,0.2)', linewidth=2, linecolor='#003D7F', tickfont={
                     'size': 12, 'color': '#003D7F'}, showgrid=True, showline=True)

    for i in range(nr_ch):
        fig['layout'].update({'xaxis{}'.format(i+1): dict(title=names[i])})
        fig['layout'].update(
            {'xaxis{}'.format(nr_ch+i+1): dict(title=names[i])})
    return fig


def graph_conf_mat(y_true, y_pred, class_names):

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    df = pd.DataFrame(np.round(cm, 2), columns=class_names, index=class_names)
    return dbc.Table.from_dataframe(df, striped=False, bordered=False, hover=True, index=True, responsive=True, size="sm", color="info", style={'color': '#003D7F', 'font-size': 14})


# START THE MAIN APP
app = dash.Dash(__name__, external_stylesheets=[
                dbc.themes.SPACELAB], suppress_callback_exceptions=True)


# background for the lower row
backgrd = html.Div(
    dbc.Container(style={"padding": "0", "margin": "0", "width": "100%", "height": "300px", "border": "0px solid #308fe3",
                         "background-image": "url(https://previews.123rf.com/images/gonin/gonin1710/gonin171000004/87977156-blue-white-gradient-hexagons-turned-abstract-background-with-geometrical-elements-modern-3d-renderin.jpg)",
                         'opacity': '0.15', 'filter': 'blur(20px)', "background-size": "100% 100%"},
                  fluid=True,
                  class_name="fixed-bottom",
                  ))


# lower row (contains all learning graphs and informations + spectrums and histograms)
conf_matrix_contents = html.Div(children=graph_conf_mat(np.array(
    [1, 2, 3]), np.array([1, 2, 3]), ['1', '2', '3']), id="table-contents")


lower_row = dbc.Nav(dbc.Container(children=[
    sliderbar,
    dbc.Container(dbc.Row(children=[html.H4("Analytics", id="lower-bar-title",
                                            style={"font-weight": "600", "padding-top": "0px"}),
                                    ]),
                    fluid=True,
                    ),
    html.Div([dbc.Row([
        dbc.Col(html.H6("Confusion Matrix",
                        style={"font-weight": "600", "margin-left": "15px"}), width={"size": 2}),
        dbc.Col(html.H6("AI Acuuracy",
                        style={"font-weight": "600", "margin-left": "0px"}), width={"size": 1, "offset": 0}),
        dbc.Col(html.H6("Power Spectrums",
                        style={"font-weight": "600", "margin-left": "65px"}), width={"size": 2, "offset": 0}),
        dbc.Col(html.H6("Amplitude Histograms",
                        style={"font-weight": "600", "margin-left": "120px"}), width={"size": 3, "offset": 2}),
    ]),
    ]),
    dbc.Nav(dbc.Container(dbc.Row([
        dbc.Container(dbc.Col(conf_matrix_contents),
            style={"width": "220px", "height": "150px", "padding": "0px"}),

        dbc.Container(dbc.Col(dcc.Graph(id="accuracy", figure=graph_ai_metric(data=np.array(params["AI_accuracy"])),
                                        responsive=True, style={"width": "200px", "height": "150px"}, config={
            'displayModeBar': False,
            'displaylogo': False,
            'scrollZoom': False,
            'editable': False,
            'showLink': False,
        })),
            style={"width": "200px", "height": "150px"}),

        dbc.Container(dbc.Col(dcc.Graph(id="hist-graphs", responsive=True,
                                        style={"width": "1350px", "height": "130px"}, config={
                                            'displayModeBar': True,
                                            'modeBarButtonsToRemove': ['zoomin', 'zoomout', 'zoom', 'pan', 'select', 'lasso2d', 'autoscale'],
                                            'displaylogo': False,
                                            'scrollZoom': False,
                                            'editable': False,
                                            'showLink':False,
                                            'toImageButtonOptions': {
                                                'format': 'png',
                                                'filename': 'PowerSpect_Histograms',
                                                'width': 1600,
                                                'scale': 1,
                                            }})),
                      style={"width": "1350px", "height": "130px"})
    ],

    ),
        fluid=True,
        style={"margin-top": "0px"}
    ),
        fill=True),

    dbc.Container(dbc.Row(
        dbc.Input(placeholder="  Training information", id="train-info", disabled=True, size='sm',
                  style={"padding": "0", "margin": "0"})),
                  fluid=True,
                  class_name="g-0 mb-0 p-0",
                  style={"margin-top": "0px"}
                  )
],

    fluid=True,
    style={"border": "0px", "width": "100%", "height": "300px"},
),
    fill=True,
    class_name="fixed-bottom",
)

# detecting keyboard keys
my_keyboard = html.Div(Keyboard(id="keyboard"))


# define app layout using dbc container
app.layout = dbc.Container(
    html.Div([my_keyboard, navbar, inputbar, graph_bar, backgrd, lower_row]), fluid=True)


# CALLBACKS

# keyboard callback
# 1. reading keyboard keys / epoch-index / max number of possible epoch
# 2. update epoch-index / user pressed key. Both in Storage
@app.callback(
    [Output("ch", "figure"),
     Output("hist-graphs", "figure"),
     Output("epoch-sliderbar", "value"),
     Output("minus-one_epoch", "value"),
     Output("null_epoch", "value"),
     Output("plus-one_epoch", "value"),
     Output("null_epoch_act", "value"),
     ],

    [Input("keyboard", "keydown"),
     Input("keyboard", "n_keydowns"),
     Input("import-offcanvas", "is_open"),
     Input("null_epoch_act", "value"),
     Input('epoch-sliderbar', 'value')
     ]
)
def keydown(event, n_keydowns, off_canvas, score_value, slider_live_value):
    # All UI offcanvases and menus (open situation) should deactivate the keyboard
    if (not event is None) and (not off_canvas):
        # read slider saved value
        slider_saved_value = params["slider_saved_value"]
        print("section 1 keyboard")
        # It is important False off_canvas /  # only in this case enter to this section (later more keys should come here)
        pressed_key_condition = (event["key"] == "ArrowRight") or (
            event["key"] == "ArrowLeft") or score_value == 1 or score_value == 2 or score_value == 3
        if (pressed_key_condition and not off_canvas) or ((slider_live_value != slider_saved_value) and not off_canvas):
            print("section 2 keyboard")
            # read params
            epoch_index = params["epoch_index"][0]
            max_nr_epochs = params["max_possible_epochs"][0]
            score_storage = params["scoring_labels"]

            # there is change in slider value / update epoch to current slider value
            if (slider_saved_value != slider_live_value):
                print("section 3 keyboard")
                epoch_index = int(slider_live_value)

            # update figures with only left/right arrow keys
            if ((event["key"] == "ArrowRight") or (event["key"] == "ArrowLeft")):
                print("section 4 keyboard")
                # check what is user pressed key
                if (event["key"] == "ArrowRight"):
                    if epoch_index < max_nr_epochs:
                        epoch_index += 1

                elif (event["key"] == "ArrowLeft"):
                    if epoch_index > 0:
                        epoch_index -= 1

                slider_live_value = epoch_index

            # update figures with score labels
            if score_value == "1" or score_value == "2" or score_value == "3":
                print("section 5 keyboard")
                # saving score label to storage
                if not score_storage is None:
                    print("section 6 keyboard")
                    # re-scoring effect
                    if epoch_index in score_storage.keys():
                        score_storage[epoch_index] = int(score_value)
                    else:
                        score_storage = score_storage.update(
                            {epoch_index: int(score_value)})
                else:
                    score_storage = {epoch_index: int(score_value)}

                if epoch_index < max_nr_epochs:
                    epoch_index += 1

                slider_live_value = epoch_index

            # read data batch from disk / check if save_path exist
            df_mid = pd.read_json(os.path.join(
                params["temp_save_path"], str(epoch_index) + ".json"))
            data_mid = np.stack(df_mid["data"])
            ps_mid = np.stack(df_mid["spectrums"]).T
            hist_mid = np.stack(df_mid["histograms"]).T

            full_ps_hist = [ps_mid, hist_mid]
            if epoch_index == max_nr_epochs:
                print("section 7 keyboard")
                data_right = np.zeros_like(data_mid)
            else:
                data_right = np.stack(pd.read_json(os.path.join(
                    params["temp_save_path"], str(epoch_index + 1) + ".json"))["data"])

            if epoch_index == 0:
                print("section 8 keyboard")
                data_left = np.zeros_like(data_mid)
            else:
                data_left = np.stack(pd.read_json(os.path.join(
                    params["temp_save_path"], str(epoch_index - 1) + ".json"))["data"])

            # combine mid_right datasets
            full_trace = np.hstack(
                [data_left,
                    data_mid,
                    data_right])

            # call for plot functions
            fig_traces = graph_channels(
                full_trace.T, names=params["selected_channels"], s_fr=params["sampling_fr"][0])
            ps_hist_fig = graph_hs_ps(
                data=full_ps_hist, names=params["selected_channels"])
            print("The current epoch index is ", epoch_index)
            # check and update score labels (after key left/right if they exist)
            if not score_storage is None:
                print("section 9 keyboard")
                if epoch_index in score_storage.keys():
                    null_score_label = str(
                        score_storage[epoch_index])
                else:
                    null_score_label = ""

                if (epoch_index - 1) in score_storage.keys():
                    epoch_minus_one_label = str(
                        score_storage[epoch_index - 1])
                else:
                    epoch_minus_one_label = ""

                if (epoch_index + 1) in score_storage.keys():
                    epoch_plus_one_label = str(
                        score_storage[epoch_index + 1])
                else:
                    epoch_plus_one_label = ""

            else:
                null_score_label = ""
                epoch_minus_one_label = ""
                epoch_plus_one_label = ""

            # change datatype
            if not score_storage is None:
                print("section 10 keyboard")
                params["scoring_labels"] = score_storage

            # check epoch and score_storage to trigger ml train
            if (not epoch_index is None) and (not score_storage is None) and (epoch_index % 10 == 0) and (epoch_index > 0):
                print("section 11 keyboard")
                ml_trigger = {"epoch_index": [epoch_index],
                              "score_storage": score_storage,
                              "save_path": params["temp_save_path"]}
            else:
                ml_trigger = dash.no_update

            return fig_traces, ps_hist_fig, slider_live_value, epoch_minus_one_label, null_score_label, epoch_plus_one_label, ""
    else:
        # Understand condition after data is loaded vs app start
        if params["data_loaded"]:
            assert os.path.exists(os.path.join(
                params["temp_save_path"], str(0) + ".json"))
            assert os.path.exists(os.path.join(
                params["temp_save_path"], str(1) + ".json"))

            # getting data
            df_mid = pd.read_json(os.path.join(
                params["temp_save_path"], str(0) + ".json"))
            data_mid = np.stack(df_mid["data"])
            ps_mid = np.stack(df_mid["spectrums"]).T
            hist_mid = np.stack(df_mid["histograms"]).T
            full_ps_hist = [ps_mid, hist_mid]

            data_left = np.zeros_like(data_mid)

            df_right = pd.read_json(os.path.join(
                params["temp_save_path"], str(1) + ".json"))
            data_right = np.stack(df_right["data"])

            # combine mid_right datasets
            full_trace = np.hstack(
                [data_left,
                    data_mid,
                    data_right])

            print("Enjoy scoring!")
            return graph_channels(full_trace.T), graph_hs_ps(full_ps_hist), 0, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            print("Keyboard first launch")
            return graph_channels(np.zeros((1000, 1))), graph_hs_ps([np.zeros((1000, 1)), np.zeros((1000, 1))]), 0, dash.no_update, dash.no_update, dash.no_update, dash.no_update


# This part has to merge to above callback after fixing issue
# reading user input for epoch len (custom value)
@app.callback(
    Output("epoch-length-input", "value"),
    Input("epoch-length-input", "value")
)
def user_custom_epoch_length(value):
    if not value is None:
        if value.isnumeric():
            print("section 2 epoch-length")
            params["epoch_length"] = [int(value)]
            time.sleep(.1)
            print(type(params["epoch_length"]), params["epoch_length"])
            return value
        else:
            # if this is the case make input invalid @Farzin
            params["epoch_length"] = ""
            return ""
    else:
        # Default value, given by user in utils.py, when app is launching
        return str(params["epoch_length"][0])


# Parameters collapse
@ app.callback(
    Output("advparam-offcanvas", "is_open"),
    [Input("advparam-button", "n_clicks")],
    [State("advparam-offcanvas", "is_open")],
)
def toggle_adv_param_offcanvas(n, is_open):
    if n:
        return not is_open
    return is_open


# secondary execution call-back
# channels importing and loading callback
@ app.callback(
    [Output("import-offcanvas", "is_open"),
     Output("channel_def_div", "children"),
     Output("load_button", "children"),
     Output("loading-output", "children"),
     Output("load_button", "disabled"),
     Output("sampling_fr_input", "disabled"),
     Output("epoch-length-input", "disabled"),
     Output("import-offcanvas-button", "n_clicks"),
     Output("load_button", "n_clicks"),
     Output("second_execution", "children"),
     Output("internal-trigger", "max_intervals"),
     Output("internal-trigger", "n_intervals"),
     Output("epoch-sliderbar", "max")
     ],

    [Input("import-offcanvas-button", "n_clicks"),
     Input("load_button", "n_clicks"),
     Input("second_execution", "children"),
     Input("internal-trigger", "n_intervals")]
)
def toggle_import_load_offcanvas(n1, n2, secondary, self_trigger):

    if secondary == True and self_trigger == 1:
        secondary = False
        print("load button running")
        max_epoch_nr = process_input_data(path_to_file=params["input_file_path"],
                                          path_to_save=params["temp_save_path"],
                                          start_index=0,
                                          end_index=-1,
                                          epoch_len=params["epoch_length"][0],
                                          fr=params["sampling_fr"][0],
                                          channel_list=params["selected_channels"],
                                          downsample=params["downsample"][0],
                                          return_result=False)
        params["max_possible_epochs"] = [max_epoch_nr]
        params["data_loaded"] = True
        params["epoch_index"] = [0]
        print(f"Max epochs: {max_epoch_nr}")
        return False, dash.no_update, "Loaded Successfully!", "", True, True, True, 0, 0, secondary, 0, 0, max_epoch_nr

    elif n2:
        print("load button action")
        n2 = n2 - 1
        channel_children = define_channels(
            channel_name=params["initial_channels"], disabled=True, value=params["selected_channels"])
        secondary = True
        return dash.no_update, channel_children, "Loading...", dash.no_update, True, True, True, 0, 0, secondary, 1, 0, dash.no_update

    elif n1 != n2:
        n1 = n1 - 1
        print("successfuly opened import button")
        # reading only data header and updating Import button configs (path is saved as text file in os.join.path(os.getcwd, "temp_saves") + "filename.txt")
        subprocess.run("python import_path.py", shell=True)

        # read input path from text file already generated
        with open(os.path.join(params["temp_save_path"], "filename.txt"), 'r') as file:
            filename = file.read()

        # update params
        params["input_file_path"] = filename

        # start reading data header (output of the file is a dataframe)
        data_header = read_data_header(filename)

        # update params
        params["initial_channels"] = data_header["channel_names"].values[0]

        # I need to run the define_channels function
        channel_children = define_channels(
            channel_name=params["initial_channels"], disabled=False, value=[])

        # button canvas, input-data-path, save-path, channel name, save-data-header
        return True, channel_children, "Load", dash.no_update, False, False, False, n1, n2, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    else:
        print("off-canvas load app is launching")
        raise PreventUpdate


@ app.callback(
    Output("sampling_fr_input", "value"),
    [Input("sampling_fr_input", "value")]
)
def handle_sample_fr_input(value):
    if not value is None:
        if value.isnumeric():
            print("section 2 sampling frequency")
            params["sampling_fr"] = [int(value)]
            time.sleep(.1)
            return value
        else:
            params["sampling_fr"] = ""
            return ""
    else:
        print("sampling frequency app is launching")
        return str(params["sampling_fr"][0])


@ app.callback(
    Output("channel_checklist", "value"),
    Input("channel_checklist", "value")
)
def get_channel_user_selection(channels):
    # pdb.set_trace()
    if not channels is None:
        try:
            main_channel_list = params["initial_channels"]
            user_selected_indices = [i for i, e in enumerate(
                main_channel_list) if e in channels]
            user_selected_indices = [
                False if i in user_selected_indices else True for i in range(len(main_channel_list))]
        except:
            user_selected_indices = [True]

        params["selected_channels"] = channels
        params["selected_channel_indices"] = user_selected_indices
        time.sleep(.1)
        return channels
    else:
        params["selected_channels"] = []
        print("channel user selection app is launching")
        return params["selected_channels"]


# inside import button (filters)
@ app.callback(
    [Output({'type': 'ddowns', 'index': ALL}, 'disabled'),
     Output({'type': 'ddowns', 'index': ALL}, 'placeholder'),
     Output({'type': 'ddowns', 'index': ALL}, 'value'),
     Output({'type': 'ddowns', 'index': ALL}, 'style'),
     Output({'type': 'mins', 'index': ALL}, 'disabled'),
     Output({'type': 'mins', 'index': ALL}, 'placeholder'),
     Output({'type': 'mins', 'index': ALL}, 'value'),
     Output({'type': 'mins', 'index': ALL}, 'style'),
     Output({'type': 'maxes', 'index': ALL}, 'disabled'),
     Output({'type': 'maxes', 'index': ALL}, 'placeholder'),
     Output({'type': 'maxes', 'index': ALL}, 'value'),
     Output({'type': 'maxes', 'index': ALL}, 'style'), ],
    Input("channel_checklist", "value")  # we don't use it
)
def toggle_disable(null_):
    indx = params["selected_channel_indices"]
    try:
        new_placeholders_ddowns = ['N/A' if i else 'Select' for i in indx]
        values_ddowns = ['' if i else dash.no_update for i in indx]
        style_ddowns = [{'width': '110px', 'filter': 'blur(50px)', 'transition': 'all 0.3s ease-out', 'opacity': '0'} if i else {
            'width': '110px', 'filter': 'blur(0px)', 'transition': 'all 0.3s ease-in', 'opacity': '100'} for i in indx]
        # in case we decide to change placeholders in the future
        new_placeholders_mins = ['Min' if i else 'Min' for i in indx]
        values_mins = ['' if i else dash.no_update for i in indx]
        style_mins = [{'width': '80px', 'filter': 'blur(50px)', 'transition': 'all 0.3s ease-out', 'opacity': '0'} if i else {
            'width': '80px', 'filter': 'blur(0px)', 'transition': 'all 0.3s ease-in', 'opacity': '100'} for i in indx]
        # in case we decide to change placeholders in the future
        new_placeholders_maxes = ['Max' if i else 'Max' for i in indx]
        values_maxes = ['' if i else dash.no_update for i in indx]
        style_maxes = [{'width': '80px', 'filter': 'blur(50px)', 'transition': 'all 0.3s ease-out', 'opacity': '0'} if i else {
            'width': '80px', 'filter': 'blur(0px)', 'transition': 'all 0.3s ease-in', 'opacity': '100'} for i in indx]
    except:
        new_placeholders_ddowns = ''
        values_ddowns = ''
        new_placeholders_mins = ''
        values_mins = ''
        new_placeholders_maxes = ''
        values_maxes = ''
    return indx, new_placeholders_ddowns, values_ddowns, style_ddowns, indx, new_placeholders_mins, values_mins, style_mins, indx, new_placeholders_maxes, values_maxes, style_maxes


# training ML
@ app.callback(
    [Output("accuracy", "figure"),
     Output("table-contents", "children")],

    Input("epoch-sliderbar", "value")
)
def train_indicator(live_slider):

    score_storage = params["scoring_labels"]
    epoch_index = params["epoch_index"][0]
    if not epoch_index is None:
        print("section 1 train indicator")
        # check if there is scoring available
        if (not score_storage is None) and (len(score_storage.keys()) > 10) and (len(score_storage.keys()) % 10 == 0):

            # check recorded class distribution
            rec_class = np.unique(list(score_storage.values()))
            print("section 2 train indicator")
            # initializing features vector and labels
            features = []
            labels = []
            # concatinate data
            for epoch in score_storage.keys():
                try:
                    df_mid = []
                    df_mid = pd.read_json(os.path.join(
                        params["temp_save_path"], str(epoch_index) + ".json"))
                    ps_mid = np.stack(df_mid["spectrums"]).T
                    hist_mid = np.stack(df_mid["histograms"]).T

                    # concatinating features
                    features.append(np.vstack([ps_mid, hist_mid]))

                    # get labels and cat them
                    labels.append(score_storage[epoch])
                except:
                    print(
                        f"Dataset for epoch #{epoch} is not found. Ignoring this epoch for training.")

            # split train test
            features = np.stack(features)

            # get features dimensions
            n, p1, p2 = features.shape

            # reshape features
            features = features.reshape(-1, p1*p2)

            # split
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.3, random_state=42)

            # start training (at the moment only using XGBoost)
            start_time = time.time()
            full_study, best_classifier = Classifier(Xtrain=X_train,
                                                     ytrain=y_train,
                                                     Xtest=X_test,
                                                     ytest=y_test).run_xgboost(n_trials=20)
            print(
                f'execution time: {np.rint(time.time() - start_time)} seconds')

            # updating AI-Accuracy vector
            params["AI_accuracy"] = np.hstack(
                [params["AI_accuracy"], [full_study.best_value]])

            # get best classifier
            the_classifier = XGBClassifier(
                **full_study.best_trial.params).fit(X_train, y_train)

            # get confusion matrix
            conf_df = graph_conf_mat(y_test, the_classifier.predict(
                X_test), [str(name) for name in np.sort(np.unique(y_test))])

            return graph_ai_metric(data=params["AI_accuracy"]), conf_df
        else:
            print(
                "Score storage is empty or doesn't satisfy proper epoch number. Training canceled!")
            raise PreventUpdate
    else:
        print("Train app is launching")
        raise PreventUpdate


# save button (remain)
"""
@ app.callback(
    Output("save-button", "children"),

    [Input("save-button", "n_clicks"),
     Input("input-file-loc", "data"),
     Input("scoring-labels", "data")]
)
def save_button(n_clicks, input_data_loc, scoring_results):
    # test
    if n_clicks:
        # first create a folder or make sure the folder exist
        save_path = os.path.join(os.path.split(
            input_data_loc)[0], "SleezyResults")
        os.makedirs(save_path, exist_ok=True)

        # saving scoring results
        #   1. reading as pandas dataframe
        scoring_results = pd.read_json(scoring_results)

        #   2. saving in any suitable format
        scoring_results.to_json(os.path.join(save_path, "score_results.json"))

        scoring_results.to_csv(os.path.join(
            save_path, "score_results.csv"), index=False)

        return "Save"
    return "Save"
"""

# run app if it get called
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    app.run_server(debug=True, threaded=True)
