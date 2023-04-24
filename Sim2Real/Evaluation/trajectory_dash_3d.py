import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State



#trajectory Points
#TODO: Replace with trajectores from csv
trajectory = np.array([
    [0.14964543, 0.51980054, 0.34284654],
    [0.15218042, 0.51606137, 0.34318519],
    [0.15552515, 0.52252108, 0.34451479],
    [0.16411068, 0.52975202, 0.34508801],
    [0.16643231, 0.53591859, 0.34596759],
    [0.16485478, 0.53941351, 0.3404704 ],
    [0.17190547, 0.54124993, 0.33658996],
    [0.17293172, 0.55146658, 0.33927181],
    [0.17038222, 0.55280972, 0.33981764],
    [0.17449835, 0.55258667, 0.34343454],
])


# Extract x, y, and z arrays from the trajectory
x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

# Create the initial figure with the 3D curve and the moving point
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Curve'))
fig.add_trace(go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode='markers', marker=dict(size=10, color='red'), name='Moving Point'))
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout with the graph and buttons
app.layout = html.Div([
    dcc.Graph(id='graph', figure=fig, style={'height': '80vh', 'width': '80vw'}),
    html.Button('Play', id='play-button', n_clicks=0),
    html.Button('Pause', id='pause-button', n_clicks=0),
    dcc.Interval(id='interval', interval=500), # speed in which the red point moves
    dcc.Store(id='n_intervals', data=0),
    dcc.Store(id='is_playing', data=True)
])

# Callback to toggle play/pause
@app.callback(
    Output('is_playing', 'data'),
    [Input('play-button', 'n_clicks'), Input('pause-button', 'n_clicks')],
    [State('is_playing', 'data')]
)
def toggle_play_pause(play_clicks, pause_clicks, is_playing):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_playing
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'play-button':
        return True
    elif button_id == 'pause-button':
        return False

# Callback to update the moving point
@app.callback(
    [Output('graph', 'figure'), Output('n_intervals', 'data')],
    [Input('interval', 'n_intervals')],
    [State('graph', 'figure'), State('is_playing', 'data'), State('n_intervals', 'data')]
)
def update_point(_, figure, is_playing, n_intervals):
    if not is_playing:
        return figure, n_intervals

    if n_intervals < len(x):
        figure['data'][1].update(x=[x[n_intervals]], y=[y[n_intervals]], z=[z[n_intervals]])
        n_intervals += 1
    return figure, n_intervals

if __name__ == '__main__':
    app.run_server(debug=True)