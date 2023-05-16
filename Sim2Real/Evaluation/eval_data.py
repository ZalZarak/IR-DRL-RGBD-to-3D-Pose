import dash
import numpy as np
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.graph_objs as go
import load_csv
from dash import dash_table
from dash.dependencies import Input,Output

#sim time = execution time

#TODO:
#Start und Goal markieren
#Computation time statt planning time, computation time farbe des planners
# 2. Chart, Steps als Barchart

#DRL = Green,
#RRT = red, 
#PRM = blue
colors = ['green', 'red', 'blue']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# load the different csv_data 
# TODO: Right now its hardcoded, this needs to change
csv_directory = csv_directory = "/home/moga/Desktop/IR-DRL/Sim2Real/Evaluation/CSV"

csv_filepaths = load_csv.get_csv_filepaths(csv_directory)


csv_PRM = load_csv.load_csv_data(csv_filepaths[0])
csv_RRT = load_csv.load_csv_data(csv_filepaths[1])
csv_DRL = load_csv.load_csv_data(csv_filepaths[2])

def count_number_of_episodes(csv_data):
     # Array erstellen, in dem für jede indexstelle die anzahl der jeweiligen Zahlen i+1 gespeichert wird
    episodes = np.array([row["episodes"]for row in csv_data])
    max_number = int(max(episodes))
    number_array = [0] * max_number

    for episode in episodes: 
         number_array[int(episode) -1] += 1

    return number_array

def planning_execution_average(csv_data,mode):
    #Hole die row aus der jeweiligen csv data und berechne die Average time für jede episode
    #[26, 31, 21, 33, 22, 21, 33, 23, 21, 22]
    
    num_episodes = count_number_of_episodes(csv_data)
    
    sim_time = np.array([row["sim_time"]for row in csv_data])
    cpu_time_steps = np.array([row["cpu_time_steps"]for row in csv_data])
    cpu_time_full = np.array([row["cpu_time_full"]for row in csv_data])
    
    lower_bound = [None for _ in range(len(num_episodes))]
    upper_bound = [None for _ in range(len(num_episodes))]
    lower_bound[0] = 0
    upper_bound[0] = num_episodes[0] -1
    for i in range(1,len(num_episodes)):
        lower_bound[i] = upper_bound[i-1]
        upper_bound[i] = upper_bound[i-1] + num_episodes[i]
       
    #TODO: different modes for RRT, DRL or PRM and hybrid
    
    computation_time_per_episode = [None for _ in range(len(num_episodes))]
    exec_time_per_episode = [None for _ in range(len(num_episodes))]
    # DRL
    if (mode == 1):
        #letzter Eintrag letzter eintrag cputime_steps
        computation_time_per_episode[0] = cpu_time_steps[upper_bound[0]]
        for i in range(len(num_episodes)):
            computation_time_per_episode[i] = cpu_time_steps[upper_bound[i]]
            exec_time_per_episode[i] = sim_time[upper_bound[i]]
    #RRT and PRM
    if (mode == 2):
        #CPUtime_full - cputime_steps 
        for i in range(len(num_episodes)):
            computation_time_per_episode[i] = cpu_time_full[upper_bound[i]] - cpu_time_steps[upper_bound[i]]
            exec_time_per_episode[i] = sim_time[upper_bound[i]]

    return computation_time_per_episode, exec_time_per_episode


plan_DRL, exec_DRL = planning_execution_average(csv_DRL,1)
plan_RRT, exec_RRT = planning_execution_average(csv_RRT,2)
plan_PRM, exec_PRM = planning_execution_average(csv_PRM,2)

# Planning execution time
planning_time_DRL_1  =plan_DRL[0] 
execution_time_DRL_1 = exec_DRL[0]
planning = [plan_DRL[0], plan_RRT[0], plan_PRM[0]]
execution = [exec_DRL[0], exec_RRT[0], exec_PRM[0]]
planning_2 = [plan_DRL[1], plan_RRT[1], plan_PRM[1]]
execution_2 = [exec_DRL[1], exec_RRT[1], exec_PRM[1]]



#shaking part
shaking_DRL = np.array([row["shaking_ur5_1"] for row in csv_DRL])
steps_DRL = list(range(1, 200))
steps_row = np.array([row[""] for row in csv_DRL])
steps_DRL = list(range(1,int(steps_row[-1])))


#distance_to_obstacle
distance_to_obstacle = [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
distance_obst_DRL = np.array([row["ur5_1_closestObstDistance_robot"]for row in csv_DRL])
distance_obst_RRT = np.array([row["ur5_1_closestObstDistance_robot"]for row in csv_RRT])



# Create the dashboard layout
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col([
            html.H1('Evaluation', style={'textAlign': 'center', 'font-family': 'Arial, sans-serif'}),
        ], width=12),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H3('Planning and Execution Time', style={'textAlign': 'center', 'font-family': 'Arial, sans-serif'}),
            dcc.Graph(
                id='planning-execution-time',
                figure={
                    'data': [
                        go.Bar(x=list(range(1, 10)), y=planning, name='Planning Time'),
                        go.Bar(x=list(range(1, 10)), y=execution, name='Execution Time')
                    ],
                    'layout': go.Layout(barmode='stack', xaxis={'title': 'Episode'}, yaxis={'title': 'Time'})
                }
            ),
            dcc.Dropdown(
                id='planning-execution-time-dropdown',
                options=[
                    {'label': 'Episode 1', 'value': 'Episode 1'},
                    {'label': 'Episode 2', 'value': 'Episode 2'},
                    {'label': 'Episode 3', 'value': 'Episode 3'},
                    {'label': 'Episode 4', 'value': 'Episode 4'},
                    {'label': 'Episode 5', 'value': 'Episode 5'},
                    {'label': 'Episode 6', 'value': 'Episode 6'},
                    {'label': 'Episode 7', 'value': 'Episode 7'},
                    {'label': 'Episode 8', 'value': 'Episode 8'},
                    {'label': 'Episode 9', 'value': 'Episode 9'},
                    {'label': 'Episode 10', 'value': 'Episode 10'}
                ],
                value='Episode 1',
                clearable=False,
                style={'width': '100%', 'font-family': 'Arial, sans-serif'}
            ),
        ], width=4),
        dbc.Col([
            html.H3('Shaking', style={'textAlign': 'center', 'font-family': 'Arial, sans-serif'}),
            dcc.Graph(
                id='shaking',
                figure={
                    'data': [
                        go.Scatter(x=steps_DRL, y=shaking_DRL, mode='lines+markers', name='DRL')
                    ],
                    'layout': go.Layout(xaxis={'title': 'Steps'}, yaxis={'title': 'Shaking'}, showlegend=True)
                }
            ),
            dcc.Dropdown(
                id='shaking-dropdown',
                options=[
                    {'label': 'Episode 1', 'value': 'Episode 1'},
                    {'label': 'Episode 2', 'value': 'Episode 2'},
                    {'label': 'Episode 3', 'value': 'Episode 3'},
                    
                ],
                value='Episode 1',
                clearable=False,
                style={'width': '100%', 'font-family': 'Arial, sans-serif'}
            ),
        ], width=4),
                dbc.Col([
            html.H3('Distance to Obstacle', style={'textAlign': 'center', 'font-family': 'Arial, sans-serif'}),
            dcc.Graph(
                id='distance-to-obstacle',
                figure={
                    'data': [
                        go.Scatter(x=steps_DRL, y=distance_to_obstacle, mode='lines+markers', name='Distance')
                    ],
                    'layout': go.Layout(
                        xaxis={'title': 'Steps'},
                        yaxis={'title': 'Distance'},
                        shapes=[{
                            'type': 'line',
                            'x0': steps_DRL[0],
                            'x1': steps_DRL[-1],
                            'y0': 0.5,
                            'y1': 0.5,
                            'yref': 'y',
                            'xref': 'x',
                            'line': {'color': 'red', 'width': 1, 'dash': 'dot'}
                        }]
                    )
                }
            ),
            dcc.Dropdown(
                id='distance-to-obstacle-dropdown',
                options=[
                    {'label': 'Placeholder 1', 'value': 'Placeholder 1'},
                    {'label': 'Placeholder 2', 'value': 'Placeholder 2'}
                ],
                value='Placeholder 1',
                clearable=False,
                style={'width': '100%', 'font-family': 'Arial, sans-serif'}
            ),
        ], width=4),
       dbc.Col([
            html.H3('Radar Chart', style={'textAlign': 'center', 'font-family': 'Arial, sans-serif'}),
            dcc.Graph(
                id='radar-chart',
                figure={
                    'data': [
                        go.Scatterpolar(
                            r=[0.5, 0.3, 0.9, 0.5],
                            theta=['smoothness', 'collision','number of steps', 'execution time'],
                            fill='toself',
                            name='Radar Chart'
                        )
                    ],
                    'layout': go.Layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1])
                        ),
                        showlegend=False
                    )
                }
            ),
            dcc.Dropdown(
                id='radar-chart-dropdown',
                options=[
                    {'label': 'Placeholder 1', 'value': 'Placeholder 1'},
                    {'label': 'Placeholder 2', 'value': 'Placeholder 2'}
                ],
                value='Placeholder 1',
                clearable=False,
                style={'width': '100%', 'font-family': 'Arial, sans-serif'}
            ),
        ], width=4),
        
    ]),
])


@app.callback(
    Output('planning-execution-time', 'figure'),
    [Input('planning-execution-time-dropdown', 'value')]
)
def update_planning_execution_chart(episode):
    if episode == 'Episode 1':
        planning_values = planning
        execution_values = execution
    elif episode == 'Episode 2':
        planning_values = planning_2
        execution_values = execution_2
    

    return {
        'data': [
            go.Bar(x=list(range(1, 4)), y=planning_values, name='Planning Time'),
            go.Bar(x=list(range(1, 4)), y=execution_values, name='Execution Time')
        ],
        'layout': go.Layout(barmode='stack', xaxis={'title': 'Episode'}, yaxis={'title': 'Time'})
    }

if __name__ == '__main__':
    app.run_server(debug=True)