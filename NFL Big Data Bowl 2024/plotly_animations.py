#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data
import pandas as pd
import numpy as np
import warnings

# Visualizations
import matplotlib.pyplot as plt
import plotly.graph_objects as go

warnings.filterwarnings("ignore") 

"""
Main Footlball Play Animation
"""

# Function to draw the football field
def draw_field(home_team, away_team):
    field = go.Figure()

    # Add green background for field and endzones
    field.add_trace(go.Scatter(x=[0, 120], y=[0, 53.3], mode='markers', marker=dict(size=1, color='green')))
    field.add_shape(type="rect", x0=0, y0=0, x1=10, y1=53.3, line=dict(color="yellow"), fillcolor="black")
    field.add_shape(type="rect", x0=110, y0=0, x1=120, y1=53.3, line=dict(color="yellow"), fillcolor="red")

    # Add yard lines & labels, avoiding the endzones
    for i in range(20, 110, 10):
        field.add_shape(type="line", x0=i, y0=0, x1=i, y1=53.3, line=dict(color="white", width=2))
        label = (i // 10) * 10  
        
        # (I know, you don't have to say it, but this was the only way I could get it to work properly right now)
        if i == 20:  
            label = '10'
        elif i == 30:
            label = '20'
        elif i == 40:
            label = '30'
        elif i == 50:
            label = '40'
        elif i == 60:
            label = '50'
        elif i == 70:
            label = '40'
        elif i == 80:
            label = '30'
        elif i == 90:
            label = '20'
        elif i == 100:
            label = '10'
        
        field.add_annotation(x=i, y=2, text=label, showarrow=False, font=dict(color="white", size=14))

    # Add team names to the endzones, rotated vertically
    field.add_annotation(x=5, y=26.65, text=away_team, showarrow=False,
                         font=dict(color="yellow", size=24),
                         xanchor="center", yanchor="middle", textangle=-90)
    field.add_annotation(x=115, y=26.65, text=home_team, showarrow=False,
                         font=dict(color="black", size=24),
                         xanchor="center", yanchor="middle", textangle=-270)

    # Set field size, hide axis lines, labels, and legend
    field.update_layout(
        xaxis=dict(showgrid=False, range=[0, 120], zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, range=[0, 53.3], zeroline=False, showticklabels=False),
        plot_bgcolor='green',
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0, pad=0)
    )
    return field

# Probability Density Effect
def add_density_effect(frame_data, defensive_positions):
    shapes = []
    for index, row in frame_data.iterrows():
        if row['position_numeric'] in defensive_positions:
            
            # The larger the tackle_probability, the larger the circle
            size = row['tackle_probability'] * 35 
            
            # Increase opacity as probability increases
            opacity = max(0.2, min(row['tackle_probability'], 1))
            
            shapes.append(go.Scatter(
                x=[row['x']],
                y=[row['y']],
                mode='markers',
                marker=dict(size=size, color='black', opacity=opacity),
                hoverinfo='skip'  # 'skip' is more appropriate than 'none' for hoverinfo
            ))
    return shapes

# Define a tackle opportunity window threshold around the ball carrier
def add_tow_circle(frame_data, ballCarrierId):
    
    # Generate a circle around the ball carrier's position with a radius of 2 yards
    ball_carrier_row = frame_data[frame_data['nflId'] == ballCarrierId].iloc[0]
    tow_circle = go.Scatter(
        x=[ball_carrier_row['x']],
        y=[ball_carrier_row['y']],
        mode='markers+text', 
        text='',  # we could add text like 'TOW' here if we want
        marker=dict(size=24,  
                    color='yellow',
                    opacity=0.35),
        hoverinfo='skip',
        showlegend=False
    )
    return tow_circle

# Initialize the football field figure with team names
field = draw_field(home_team="Atalnta Falcons", away_team="New Orleans Saints") 

# Function to add player markers for a single frame
def add_player_markers(frame_data, defensive_positions):
    
    # List to store the Scatter objects for the current frame
    scatters = []
    for index, row in frame_data.iterrows():
        if row['nflId'] == row['ballCarrierId']:
            color = 'yellow'  # Ball carrier color
        elif row['position_numeric'] in defensive_positions:
            color = 'black'  # defense color
        else:
            color = 'red'  # offense color
        size = 8 if row['nflId'] != row['ballCarrierId'] else 10 
        scatters.append(go.Scatter(x=[row['x']], y=[row['y']], mode='markers', marker=dict(size=size, color=color)))
    return scatters

ballCarrierId = final_play_df['ballCarrierId'].iloc[0] 

# Create frames for each time step in the play
frames = []
for frame_id in sorted(final_play_df['frameId'].unique()):
    frame_data = final_play_df[final_play_df['frameId'] == frame_id]
    frame_scatters = add_player_markers(frame_data, defensive_positions)
    frame_densities = add_density_effect(frame_data, defensive_positions)
    frame_tow_circle = add_tow_circle(frame_data, ballCarrierId)
    frames.append(go.Frame(data=frame_scatters + frame_densities + [frame_tow_circle], name=str(frame_id)))

# Set the frames of the animation
field.frames = frames

# Add the initial frame data (player markers and density effect only)
initial_frame_data = add_player_markers(final_play_df[final_play_df['frameId'] == final_play_df['frameId'].min()], 
                                        defensive_positions)
initial_density_data = add_density_effect(final_play_df[final_play_df['frameId'] == final_play_df['frameId'].min()], 
                                          defensive_positions)
field.add_traces(initial_frame_data + initial_density_data)

# Layout updates for the animation to make it smoother, this can be highly customizable
field.update_layout(
    updatemenus=[
        {
            "type": "buttons",
            "showactive": False,
            "y": .065,
            "x": 0.9,
            "xanchor": "center",
            "yanchor": "top",
            "pad": {"t": 45, "r": 10},
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 50, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 25, "easing": "linear"},
                        },
                    ],
                }
            ],
        }
    ],
    sliders=[
        {
            "steps": [
                {
                    "method": "animate",
                    "args": [
                        [f"{frame}"],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 50, "redraw": True},
                            "transition": {"duration": 25},
                        },
                    ],
                    "label": f"{frame}",
                }
                for frame in final_play_df['frameId'].unique()
            ],
            "transition": {"duration": 10},
            "x": 0,
            "y": 0,
            "currentvalue": {
                "font": {"size": 12},
                "prefix": "Frame: ",
                "visible": True,
                "xanchor": "center",
            },
            "len": 1.0,
        }
    ],
)

field.show()

"""
Time Series Analysis Animation
"""

# Create a mapping of nflId to player names
player_names = dict(zip(players['nflId'], players['displayName']))

defensive_positions = [4, 5, 6, 7, 9, 10, 11, 14, 15, 18]  

# Filter to get only defensive players
defenders_df = final_play_df[final_play_df['position_numeric'].isin(defensive_positions)]

# Initialize the figure
fig = go.Figure()

# Set up fixed axes ranges
max_frame = defenders_df['frameId'].max()
max_probability = defenders_df['tackle_probability'].max()

fig.update_layout(
    xaxis=dict(range=[0, max_frame], autorange=False),
    yaxis=dict(range=[0, max_probability], autorange=False),
    title="Tackle Expectation Sequencing",
    xaxis_title="Time",
    yaxis_title="P(Tackle)"
)

fig.update_layout(
    plot_bgcolor='black',  
    paper_bgcolor='green',  
    font=dict(color='white'), 
    title=dict(x=0.5, y=0.86, xanchor='center', font=dict(size=20)),  
    xaxis=dict(color='white'),  
    yaxis=dict(color='white'), 
)


# Add initial empty traces for each defender
for nflId in defenders_df['nflId'].unique():
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode='lines',
            name=player_names.get(nflId, f'Defender {nflId}') #
        )
    )

# Function to get the data for each defender up to the current frame
def get_defender_data_up_to_frame(defenders_df, nflId, current_frame):
    return defenders_df[(defenders_df['nflId'] == nflId) & (defenders_df['frameId'] <= current_frame)]

# Creating frames for the animation
frames = []
for frame_id in sorted(defenders_df['frameId'].unique()):
    frame_traces = []
    
    for nflId in defenders_df['nflId'].unique():
        defender_data = get_defender_data_up_to_frame(defenders_df, nflId, frame_id)
        frame_traces.append(
            go.Scatter(
                x=defender_data['frameId'],
                y=defender_data['tackle_probability'],
                mode='lines'
            )
        )
    
    frames.append(go.Frame(data=frame_traces, name=str(frame_id)))

fig.frames = frames

# Add play control buttons
fig.update_layout(
    updatemenus=[
        {
            "type": "buttons",
            "direction": "left",
            "x": 0.95,  
            "y": -0.5,  
            "xanchor": "left",
            "yanchor": "bottom",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, 
                             {"frame": {"duration": 50, "redraw": True}, 
                              "fromcurrent": True, 
                              "transition": {"duration": 25}}]
                }
            ]
        }
    ]
)

fig.show()

