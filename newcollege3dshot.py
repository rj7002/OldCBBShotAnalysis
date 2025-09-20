# import streamlit as st
# import requests
# import pandas as pd
# from io import StringIO

# season = st.selectbox('Select a season', (list(range(2014,2025))))
# import requests
# import pyreadr

# # URL of the RDS file
# url = f"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_pbp/play_by_play_{season}.rds"

# # Fetch the RDS file content
# response = requests.get(url)

# # Save the content to a local file
# with open('play_by_play_2025.rds', 'wb') as file:
#     file.write(response.content)

# # Read the RDS file using pyreadr
# result = pyreadr.read_r('play_by_play_2025.rds')

# # Extract the data frame (assuming it's the first element in the result dictionary)
# my_data = result[None]

# adf = pd.DataFrame(my_data)
# adf[adf['shooting_play'] == True]
# playernames = adf['text'].apply(lambda x: f'{x.split(' ')[0]} {x.split(' ')[1]}').unique()
# player = st.selectbox('Select a player',playernames)
# adf = adf[adf['text'].str.contains(f'{player} made|{player} missed')]
# st.write(adf)
# # Now `my_data` will contain the data from the RDS file

import pandas as pd
from pandas_gbq import read_gbq
import streamlit as st
import seaborn as sns
from courtCoordinates import CourtCoordinates
import plotly.express as px
import plotly.graph_objects as go  
import numpy as np
from random import randint
# Set your Google Cloud project ID
project_id = ''

# SQL query - adjust based on your needs
st.set_page_config(page_title='Old CBB Shot Analysis',page_icon='🏀')
st.title('CBB Shot Analysis (Old Version)')
name = st.text_input('Enter a D1 player name (2013-14 to 2017-18)')
name = name.title()
st.write(name)
if name:
    query = f"""
        SELECT
            event_description,event_coord_x,event_coord_y,event_type,team_basket,points_scored,shot_made,shot_type,game_clock,home_market,away_market,season,period,scheduled_date

        FROM
            `bigquery-public-data.ncaa_basketball.mbb_pbp_sr`  
        WHERE 
            event_description LIKE '%{name}%' AND (event_type = 'twopointmiss' OR event_type = 'twopointmade' OR event_type = 'threepointmiss' OR event_type = 'threepointmade')
    """

    df = read_gbq(query, project_id=project_id)
   
    if len(df) > 0:
        selectseasons = st.multiselect('Select seasons',df['season'].unique(),default=df['season'].unique()[0])
        df = df[df['season'].isin(selectseasons)]
        df = df[df['event_description'].str.contains(f'{name} makes|{name} misses')]
        # Show the results
        df.loc[df['team_basket'] == "right","event_coord_x"] = abs(1128-df["event_coord_x"])
        df.loc[df['team_basket'] == "right","event_coord_y"] = abs(600-df["event_coord_y"])

        df['event_coord_y'] = df['event_coord_y']*.85-260 
        df['event_coord_x'] = df['event_coord_x']*.75+20
        df['game_date'] = df['scheduled_date'].apply(lambda x: f'{x.date()}')
        court = CourtCoordinates('2014-15')
        court_lines_df = court.get_coordinates()
        df['shotDist'] = np.sqrt((((df['event_coord_x'])) - court.hoop_loc_x)**2 + 
                                (((df['event_coord_y'])) - court.hoop_loc_y)**2)
        df = df[df['shotDist']/10 <= 50]
        def shotmade(r):
            if 'makes' in r:
                return True
            elif 'misses' in r:
                return False
            return None
        df['shot_made'] = df['event_description'].apply(shotmade)
        st.success('Player Found')
        col1, col2 = st.columns(2)
        # st.write(court_lines_df)
        fig = px.line_3d(
                data_frame=court_lines_df,
                x='x',
                y='y',
                z='z',
                line_group='line_group_id',
                color='line_group_id',
                color_discrete_map={
                    'court': 'black',
                    'hoop': '#e47041',
                    'net': '#D3D3D3',
                    'backboard': 'gray',
                    'backboard2': 'gray',
                    'free_throw_line': 'black',
                    'hoop2':'#D3D3D3',
                    'free_throw_line2': 'black',
                    'free_throw_line3': 'black',
                    'free_throw_line4': 'black',
                    'free_throw_line5': 'black',
                }
            )
        fig.update_traces(hovertemplate=None, hoverinfo='skip', showlegend=False)
        fig.update_traces(line=dict(width=6))
        court_perimeter_bounds = np.array([[-250, 0, -0.2], [250, 0, -0.2], [250, 450, -0.2], [-250, 450, -0.2], [-250, 0, -0.2]])
        
        # Extract x, y, and z values for the mesh
        court_x = court_perimeter_bounds[:, 0]
        court_y = court_perimeter_bounds[:, 1]
        court_z = court_perimeter_bounds[:, 2]
        
        # Add a square mesh to represent the court floor at z=0
        fig.add_trace(go.Mesh3d(
            x=court_x,
            y=court_y,
            z=court_z,
            color='#d2a679',
            # opacity=0.5,
            name='Court Floor',
            hoverinfo='none',
            showscale=False
        ))
        fig.update_layout(    
            margin=dict(l=20, r=20, t=20, b=20),
            scene_aspectmode="data",
            height=600,
            scene_camera=dict(
                eye=dict(x=1.3, y=0, z=0.7)
            ),
            scene=dict(
                xaxis=dict(title='', showticklabels=False, showgrid=False),
                yaxis=dict(title='', showticklabels=False, showgrid=False),
                zaxis=dict(title='',  showticklabels=False, showgrid=False, showbackground=False, backgroundcolor='#d2a679'),
            ),
            showlegend=False,
            legend=dict(
                yanchor='top',
                y=0.05,
                x=0.2,
                xanchor='left',
                orientation='h',
                font=dict(size=15, color='gray'),
                bgcolor='rgba(0, 0, 0, 0)',
                title='',
                itemsizing='constant'
            )
        )
        # df = df[df['points_scored'] == 3]
        hover_label = df.apply(lambda row:f"""
                <b>Desc:</b> {row['event_description']}<br>
                <b>Shot Distance:</b> {row['shotDist']/12} ft<br>
                <b>Shot Type:</b> {row['shot_type']}<br>
                <b>Period:</b> {row['period']}<br>
                <b>Time:</b> {row['game_clock']}<br>
                <b>Game:</b> {row['home_market']} vs {row['away_market']}<br>
                <b>Date:</b> {row['game_date']}<br>
                <b>Season:</b> {row['season']}<br>


                """, axis=1)
        import numpy as np
        df['color'] = np.where(df['shot_made'] == False,'red','green')
        df['symbol'] = np.where(df['shot_made'] == True, 'circle', 'cross')

       
    

        if st.checkbox('Animated'):
            newdf = df.copy()
            newdf = newdf[newdf['shot_made'] == True]
            newdf = newdf[newdf['shotDist'] > 30]
            # if len(newdf) > 150: 
            #     st.error(f'Too many shots. Only showing first 150 shots.')
            #     newdf = newdf.head(150)
            # else:
            #     newdf = newdf
            if len(newdf) >= 100:
                    default = 10
            elif len(newdf) >= 75:
                default = 8
            elif len(newdf) >= 50:
                default = 5
            elif len(newdf) >= 20:
                default = 3
            else:
                default = 1
            cl1, cl2 = st.columns(2)
            with cl1:
                shotgroup = st.number_input("Number of Shots Together", min_value=1, max_value=10, step=1, value=default)
            with cl2:
                speed = st.selectbox('Speed',['Fast','Medium','Slow'])
            if speed == 'Fast':
                delay = 0.00000000000000000000001
            elif speed == 'Medium':
                delay = 150
            elif speed == 'Slow':
                delay = 200
            court_perimeter_bounds = np.array([[-250, 0, 0], [250, 0, 0], [250, 450, 0], [-250, 450, 0], [-250, 0, 0]])
            
            # Extract x, y, and z values for the mesh
            court_x = court_perimeter_bounds[:, 0]
            court_y = court_perimeter_bounds[:, 1]
            court_z = court_perimeter_bounds[:, 2]
            
            # Add a square mesh to represent the court floor at z=0
            fig.add_trace(go.Mesh3d(
                x=court_x,
                y=court_y,
                z=court_z-1,
                color='#d2a679',
                opacity=1,
                name='Court Floor',
                hoverinfo='none',
                showscale=False
            ))
            # hover_data = newdf.apply(lambda row: f"""
            #     <b>Player:</b> {row['fullName']}<br>
            #     <b>Game Date:</b> {row['gameDate']}<br>
            #     <b>Game:</b> {row['TeamName']} vs {row['OpponentName']}<br>
            #     <b>Half:</b> {row['period'][-1]}<br>
            #     <b>Time:</b> {row['clock']}<br>
            #     <b>Result:</b> {'Made' if row['success'] else 'Missed'}<br>
            #     <b>Shot Distance:</b> {row['shotDist']} ft<br>
            #     <b>Shot Type:</b> {row['actionType']} ({row['subType']})<br>
            #     <b>Shot Clock:</b> {row['shotClock']}<br>
            #     <b>Assisted by:</b> {row['assisterName']}<br>
            # """, axis=1)
        
            court_perimeter_lines = court_lines_df[court_lines_df['line_id'] == 'outside_perimeter']
            three_point_lines = court_lines_df[court_lines_df['line_id'] == 'three_point_line']
            backboard = court_lines_df[court_lines_df['line_id'] == 'backboard']
            backboard2 = court_lines_df[court_lines_df['line_id'] == 'backboard2']
            freethrow = court_lines_df[court_lines_df['line_id'] == 'free_throw_line']
            freethrow2 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line2']
            freethrow3 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line3']
            freethrow4 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line4']
            freethrow5 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line5']
            hoop = court_lines_df[court_lines_df['line_id'] == 'hoop']
            hoop2 = court_lines_df[court_lines_df['line_id'] == 'hoop2']
            
            
            
            
            
            
            
            # Add court lines to the plot (3D scatter)
            fig.add_trace(go.Scatter3d(
                x=court_perimeter_lines['x'],
                y=court_perimeter_lines['y'],
                z=np.zeros(len(court_perimeter_lines)),  # Place court lines on the floor
                mode='lines',
                line=dict(color='black', width=6),
                name="Court Perimeter",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=hoop['x'],
                y=hoop['y'],
                z=hoop['z'],  # Place 3-point line on the floor
                mode='lines',
                line=dict(color='#e47041', width=6),
                name="Hoop",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=hoop2['x'],
                y=hoop2['y'],
                z=hoop2['z'],  # Place 3-point line on the floor
                mode='lines',
                line=dict(color='#D3D3D3', width=6),
                name="Backboard",
                hoverinfo='none'
            ))
            # Add the 3-point line to the plot
            fig.add_trace(go.Scatter3d(
                x=backboard['x'],
                y=backboard['y'],
                z=backboard['z'],  # Place 3-point line on the floor
                mode='lines',
                line=dict(color='grey', width=6),
                name="Backboard",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=backboard2['x'],
                y=backboard2['y'],
                z=backboard2['z'],  # Place 3-point line on the floor
                mode='lines',
                line=dict(color='grey', width=6),
                name="Backboard",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=three_point_lines['x'],
                y=three_point_lines['y'],
                z=np.zeros(len(three_point_lines)),  # Place 3-point line on the floor
                mode='lines',
                line=dict(color='black', width=6),
                name="3-Point Line",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=freethrow['x'],
                y=freethrow['y'],
                z=np.zeros(len(freethrow)),  # Place court lines on the floor
                mode='lines',
                line=dict(color='black', width=6),
                name="Court Perimeter",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=freethrow2['x'],
                y=freethrow2['y'],
                z=np.zeros(len(freethrow2)),  # Place court lines on the floor
                mode='lines',
                line=dict(color='black', width=6),
                name="Court Perimeter",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=freethrow3['x'],
                y=freethrow3['y'],
                z=np.zeros(len(freethrow3)),  # Place court lines on the floor
                mode='lines',
                line=dict(color='black', width=6),
                name="Court Perimeter",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=freethrow4['x'],
                y=freethrow4['y'],
                z=np.zeros(len(freethrow4)),  # Place court lines on the floor
                mode='lines',
                line=dict(color='black', width=6),
                name="Court Perimeter",
                hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=freethrow5['x'],
                y=freethrow5['y'],
                z=np.zeros(len(freethrow5)),  # Place court lines on the floor
                mode='lines',
                line=dict(color='black', width=6),
                name="Court Perimeter",
                hoverinfo='none'
            ))
            x_values = []
            y_values = []
            z_values = []
            # dfmiss = df[df['SHOT_MADE_FLAG'] == 0]
            # df = df[df['SHOT_MADE_FLAG'] == 1]
            

            for index, row in newdf.iterrows():
                
                
            
                x_values.append(row['event_coord_y'])
                # Append the value from column 'x' to the list
                y_values.append(row['event_coord_x'])
                z_values.append(0)
            
            
            
            x_values2 = []
            y_values2 = []
            z_values2 = []
            import math
            for index, row in newdf.iterrows():
                # Append the value from column 'x' to the list
            
            
                x_values2.append(court.hoop_loc_x)
            
                y_values2.append(court.hoop_loc_y)
                z_values2.append(100)
            
            def calculate_distance(x1, y1, x2, y2):
                return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Function to generate arc points
            def generate_arc_points(p1, p2, apex, num_points=100):
                t = np.linspace(0, 1, num_points)
                x = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * apex[0] + t**2 * p2[0]
                y = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * apex[1] + t**2 * p2[1]
                z = (1 - t)**2 * p1[2] + 2 * (1 - t) * t * apex[2] + t**2 * p2[2]
                return x, y, z
            
            
        
            
            frames = []
            num_points = 200  # Increase this for more resolution
            segment_size = 20  # Number of points per visible segment
            
            # Function to process shots in batches
            def process_shots_in_batches(shotdf, batch_size=3):
                for batch_start in range(0, len(shotdf), batch_size):
                    batch_end = min(batch_start + batch_size, len(shotdf))
                    yield shotdf[batch_start:batch_end]
            
            # Generate frames for each batch
            for batch in process_shots_in_batches(newdf, batch_size=shotgroup):
                for t in np.linspace(0, 1, 8):  # Adjust for smoothness
                    frame_data = []
                    
                    for _, row in batch.iterrows():
                        x1, y1 = int(row['event_coord_y']), int(row['event_coord_x'])
                        x2, y2 = court.hoop_loc_x, court.hoop_loc_y
                        p2 = np.array([x1, y1, 0])
                        p1 = np.array([x2, y2, 100])
            
                        # Arc height based on shot distance
                    #      if df['shotDist'].iloc[i]/10 > 50:
                    #     h = randint(255,305)
                    # elif df['shotDist'].iloc[i]/10 > 30:
                    #     h = randint(230,250)-30
                    # elif df['shotDist'].iloc[i]/10 > 25: 
                    #     h = randint(220,230)-30
                    # elif df['shotDist'].iloc[i]/10 > 15:
                    #     h = randint(200,230)
                    # else:
                    #     h = randint(130,160)
                    
                        h = (randint(130,160) if row['shotDist'] <= 15 else
                            randint(200,230) if row['shotDist'] <= 25 else
                            randint(220,230) if row['shotDist'] <= 30 else
                            randint(230,250) if row['shotDist'] <= 50 else
                            randint(255,305))
                        apex = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), h])
                        x, y, z = generate_arc_points(p2, p1, apex, num_points)
            
                        # Calculate the start and end of the moving segment
                        total_points = len(x)
                        start_index = int(t * (total_points - segment_size))
                        end_index = start_index + segment_size
            
                        # Ensure indices are within bounds
                        start_index = max(0, start_index)
                        end_index = min(total_points, end_index)
            
                        segment_x = x[start_index:end_index]
                        segment_y = y[start_index:end_index]
                        segment_z = z[start_index:end_index]
            
                        frame_data.append(go.Scatter3d(
                            x=segment_x, y=segment_y, z=segment_z,
                            mode='lines', line=dict(width=6, color=row['color']),
                            hoverinfo='text', hovertext=row.get('hover_text', '')
                        ))

                    frames.append(go.Frame(data=frame_data))
            
            
            # Add an initial empty trace for layout
            fig.add_trace(go.Scatter3d(x=[], y=[], z=[]))
            
            # Empty frame at the end for clearing the court
            fig.add_trace(go.Scatter3d(x=[], y=[], z=[]))
            empty_frame_data = []
            for i in range(0,10):
                empty_frame_data.append(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='lines', line=dict(width=6, color='rgba(255, 0, 0, 0)')
            ))
            
            frames.append(go.Frame(data=empty_frame_data))
            
            
            # Add frames to the figure
            fig.frames = frames
            
            
            
            # Layout with animation controls
            fig.update_layout(
                updatemenus=[
                    dict(type="buttons",
                        showactive=False,
                        buttons=[
                            dict(label="Play",
                                method="animate",
                                args=[None, {"frame": {"duration": delay, "redraw": True}, "fromcurrent": True}]),
                            dict(label="Pause",
                                method="animate",
                                args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                        ])
                ],
                # # scene=dict(
                # #     # xaxis=dict(range=[-25, 25], title="X"),
                # #     # yaxis=dict(range=[-50, 50], title="Y"),
                #     zaxis=dict(range=[0, 175], title="Z"),
                # #     # aspectratio=dict(x=1, y=1, z=0.5),
                # # ),
            )
            with col1:
                st.plotly_chart(fig)
        else:
            fig.add_trace(go.Scatter3d(
                                    x=df['event_coord_y'], y=df['event_coord_x'], z=len(df)*[0],
                                    mode='markers',
                                    marker=dict(size=8, color=df['color'], opacity=0.6,symbol=df['symbol']),
                                    # name=f'Arc {i + 1}',
                                    hoverinfo='text',
                                    hovertext = hover_label
                                ))

            x_values = []
            y_values = []
            z_values = []
            for index, row in df.iterrows():
                
                
            
                x_values.append(row['event_coord_x'])
                # Append the value from column 'x' to the list
                y_values.append(row['event_coord_y'])
                z_values.append(0)
            
            
            
            x_values2 = []
            y_values2 = []
            z_values2 = []
            for index, row in df.iterrows():
                # Append the value from column 'x' to the list
            
            
                x_values2.append(court.hoop_loc_y)
            
                y_values2.append(court.hoop_loc_x)
                z_values2.append(100)
            
            import numpy as np
            import plotly.graph_objects as go
            import streamlit as st
            import math
            def calculate_distance(x1, y1, x2, y2):
                """Calculate the distance between two points (x1, y1) and (x2, y2)."""
                return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            def generate_arc_points(p1, p2, apex, num_points=100):
                """Generate points on a quadratic Bezier curve (arc) between p1 and p2 with an apex."""
                t = np.linspace(0, 1, num_points)
                x = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * apex[0] + t**2 * p2[0]
                y = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * apex[1] + t**2 * p2[1]
                z = (1 - t)**2 * p1[2] + 2 * (1 - t) * t * apex[2] + t**2 * p2[2]
                return x, y, z
            
            # Example lists of x and y coordinates
            x_coords = x_values
            y_coords = y_values
            z_value = 0  # Fixed z value
            x_coords2 = x_values2
            y_coords2 = y_values2
            z_value2 = 100
            # st.write(shotdf)
            # df = df[df['shot_made'] == True]
            # df = df[df['shotDist'] <= 80]
            for i in range(len(df)):
                
            
                # if df['SHOT_MADE_FLAG'].iloc[i] == 1:
                #     s = 'circle-open'
                #     s2 = 'circle'
                #     size = 9
                #     color = 'green'
                # else:
                #     s = 'cross'
                #     s2 = 'cross'
                #     size = 10
                #     color = 'red'
                # date_str = df['GAME_DATE'].iloc[i]
                # game_date = datetime.strptime(date_str, "%Y%m%d")
                # formatted_date = game_date.strftime("%m/%d/%Y")
                # if int(df['SECONDS_REMAINING'].iloc[i]) < 10:
                #     df['SECONDS_REMAINING'].iloc[i] = '0' + str(df['SECONDS_REMAINING'].iloc[i])
                # hovertemplate= f"Date: {formatted_date}<br>Game: {df['HTM'].iloc[i]} vs {df['VTM'].iloc[i]}<br>Result: {df['EVENT_TYPE'].iloc[i]}<br>Shot Type: {df['ACTION_TYPE'].iloc[i]}<br>Distance: {df['SHOT_DISTANCE'].iloc[i]} ft {df['SHOT_TYPE'].iloc[i]}<br>Quarter: {df['PERIOD'].iloc[i]}<br>Time: {df['MINUTES_REMAINING'].iloc[i]}:{df['SECONDS_REMAINING'].iloc[i]}"
                if (df['shotDist'].iloc[i] > 3*10) and (df['shot_made'].iloc[i] == True):
                    x1 = x_coords[i]
                    y1 = y_coords[i]
                    x2 = x_coords2[i]
                    y2 = y_coords2[i]
                    # Define the start and end points
                    p2 = np.array([x1, y1, z_value])
                    p1 = np.array([x2, y2, z_value2])
                    
                    # Apex will be above the line connecting p1 and p2
                    distance = calculate_distance(x1, y1, x2, y2)
                    if df['shotDist'].iloc[i]/10 > 50:
                        h = randint(255,305)
                    elif df['shotDist'].iloc[i]/10 > 30:
                        h = randint(230,250)-30
                    elif df['shotDist'].iloc[i]/10 > 25: 
                        h = randint(220,230)-30
                    elif df['shotDist'].iloc[i]/10 > 15:
                        h = randint(200,230)
                    else:
                        h = randint(130,160)
                

                    # h = randint(165,200)
                    # if df['shotDist'].iloc[i] < 60:
                    #     h = randint(165,200)
                    # if teamcolors:
                    #     color = dfmiss['color1'].iloc[0]
                    # else:
                    if df['shot_made'].iloc[i] == False:
                            color = 'red'
                    else:
                            color = 'green'
                    apex = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), h])  # Adjust apex height as needed
                    
                    # Generate arc points
                    row = df.iloc[i]
                    
                    # # Create the hover data string for the current row
                    # hover_label = f"""
                    # <b>Player:</b> {row['fullName']}<br>
                    # <b>Game Date:</b> {row['gameDate']}<br>
                    # <b>Game:</b> {row['TeamName']} vs {row['OpponentName']}<br>
                    # <b>Half:</b> {row['period'][-1]}<br>
                    # <b>Time:</b> {row['clock']}<br>
                    # <b>Result:</b> {'Made' if row['success'] else 'Missed'}<br>
                    # <b>Shot Distance:</b> {row['shotDist']} ft<br>
                    # <b>Shot Type:</b> {row['actionType']} ({row['subType']})<br>
                    # <b>Shot Clock:</b> {row['shotClock']}<br>
                    # <b>Assisted by:</b> {row['assisterName']}<br>
                    
                    hover_label = f"""
                    <b>Desc:</b> {row['event_description']}<br>
                    <b>Shot Distance:</b> {row['shotDist']/12} ft<br>
                    <b>Shot Type:</b> {row['shot_type']}<br>
                    <b>Period:</b> {row['period']}<br>
                    <b>Time:</b> {row['game_clock']}<br>
                    <b>Game:</b> {row['home_market']} vs {row['away_market']}<br>
                    <b>Date:</b> {row['game_date']}<br>
                    <b>Season:</b> {row['season']}<br>


                    """
                    x, y, z = generate_arc_points(p1, p2, apex)
                    fig.add_trace(go.Scatter3d(
                                x=y, y=x, z=z,
                                mode='lines',
                                line=dict(width=8,color = color),
                                opacity =0.5,
                                # name=f'Arc {i + 1}',
                                hoverinfo='text',
                                hovertext=hover_label
                            ))
            with col1:
                st.plotly_chart(fig,use_container_width=True)


        
        if st.checkbox('FG%'):
            x_bins = np.linspace(-270, 270, 20)  # 30 bins along X axis (basketball court length)
            y_bins = np.linspace(-10, 450, 10)   # 20 bins along Y axis (basketball court width)
            
            # Create 2D histograms: one for shot attempts (total shots) and one for made shots
            shot_attempts, x_edges, y_edges = np.histogram2d(df['event_coord_y'], df['event_coord_x'], bins=[x_bins, y_bins])
            made_shots, _, _ = np.histogram2d(df['event_coord_y'][df['shot_made'] == True], df['event_coord_x'][df['shot_made'] == True], bins=[x_bins, y_bins])
            
            # Calculate the Field Goal Percentage (FG%) for each bin
            fg_percentage = np.divide(made_shots, shot_attempts, where=shot_attempts != 0) * 100  # Avoid division by zero
            
            # Normalize FG% for color mapping (to make sure it stays between 0 and 100)
            fg_percentage_normalized = np.clip(fg_percentage, 0, 100)  # Clamp FG% between 0 and 100
            
            # Calculate the center of each bin for plotting (bin centers)
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            
            # Create a meshgrid of X and Y centers for 3D plotting
            X, Y = np.meshgrid(x_centers, y_centers)
            
            # Create hovertext to show FG% for each region
            hovertext = np.array([f'FG%: {fg}%' for fg in fg_percentage.flatten()]).reshape(fg_percentage.shape)

            
            # Create the 3D surface plot
            z_max = 100  # Replace with the desired limit
            Z = shot_attempts.T
            Z2 = Z *5
            Z2 = np.minimum(Z2, z_max)
            fig = go.Figure(data=go.Surface(
                z=Z2,  # Shot density (number of shots) as the Z-axis
                x=X,  # X values (bin centers)
                y=Y,  # Y values (bin centers)
                
                # Surface color based on Field Goal Percentage (FG%)
                surfacecolor=fg_percentage.T,  # Use FG% as the surface color
                
                colorscale='hot',  # Color scale based on FG% (you can change this to any scale)
                cmin=0,  # Minimum FG% for color scale
                cmax=100,  # Maximum FG% for color scale
                colorbar=dict(title='Field Goal %'),  # Color bar label
                showscale=False,  # Show the color scale/legend
                hoverinfo='none',  # Show text on hover
                # hovertext=hovertext  # Attach the hovertext showing FG%
            ))
        else:
            x_bins = np.linspace(-270, 270, 20)  # 30 bins along X axis (basketball court length)
            y_bins = np.linspace(-10, 450, 10)   # 20 bins along Y axis (basketball court width)

            
            # Create 2D histogram to get shot density
            shot_density, x_edges, y_edges = np.histogram2d(df['event_coord_y'], df['event_coord_x'], bins=[x_bins, y_bins])
            
            # Calculate the center of each bin for plotting
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            
            # Create a meshgrid of X and Y centers for 3D plotting
            X, Y = np.meshgrid(x_centers, y_centers)
            Z = shot_density.T  # Transpose to match the correct orientation for plotting
            Z2 = Z*5
            z_max = 100  # Replace with the desired limit

            # Apply the limit to Z values
            Z2 = np.minimum(Z2, z_max)
            # Plot 3D shot density
            hovertext = np.array([f'Shots: {z}' for z in Z.flatten()]).reshape(Z.shape)
            fig = go.Figure(data=go.Surface(
                z=Z2,
                x=X,
                y=Y,
                colorscale='hot',  # You can choose different color scales
                colorbar=dict(title='Shot Density'),
                showscale=False  # Hide the color bar/legend
                ,hoverinfo='text',
                hovertext=hovertext
            ))

        court_perimeter_lines = court_lines_df[court_lines_df['line_id'] == 'outside_perimeter']
        three_point_lines = court_lines_df[court_lines_df['line_id'] == 'three_point_line']
        backboard = court_lines_df[court_lines_df['line_id'] == 'backboard']
        backboard2 = court_lines_df[court_lines_df['line_id'] == 'backboard2']
        freethrow = court_lines_df[court_lines_df['line_id'] == 'free_throw_line']
        freethrow2 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line2']
        freethrow3 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line3']
        freethrow4 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line4']
        freethrow5 = court_lines_df[court_lines_df['line_id'] == 'free_throw_line5']
        hoop = court_lines_df[court_lines_df['line_id'] == 'hoop']
        hoop2 = court_lines_df[court_lines_df['line_id'] == 'hoop2']
        
        
        
        
        
        
        
        # Add court lines to the plot (3D scatter)
        fig.add_trace(go.Scatter3d(
            x=court_perimeter_lines['x'],
            y=court_perimeter_lines['y'],
            z=np.zeros(len(court_perimeter_lines)),  # Place court lines on the floor
            mode='lines',
            line=dict(color='white', width=6),
            name="Court Perimeter",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=hoop['x'],
            y=hoop['y'],
            z=hoop['z'],  # Place 3-point line on the floor
            mode='lines',
            line=dict(color='#e47041', width=6),
            name="Hoop",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=hoop2['x'],
            y=hoop2['y'],
            z=hoop2['z'],  # Place 3-point line on the floor
            mode='lines',
            line=dict(color='#D3D3D3', width=6),
            name="Backboard",
            hoverinfo='none'
        ))
        # Add the 3-point line to the plot
        fig.add_trace(go.Scatter3d(
            x=backboard['x'],
            y=backboard['y'],
            z=backboard['z'],  # Place 3-point line on the floor
            mode='lines',
            line=dict(color='grey', width=6),
            name="Backboard",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=backboard2['x'],
            y=backboard2['y'],
            z=backboard2['z'],  # Place 3-point line on the floor
            mode='lines',
            line=dict(color='grey', width=6),
            name="Backboard",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=three_point_lines['x'],
            y=three_point_lines['y'],
            z=np.zeros(len(three_point_lines)),  # Place 3-point line on the floor
            mode='lines',
            line=dict(color='white', width=6),
            name="3-Point Line",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=freethrow['x'],
            y=freethrow['y'],
            z=np.zeros(len(freethrow)),  # Place court lines on the floor
            mode='lines',
            line=dict(color='white', width=6),
            name="Court Perimeter",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=freethrow2['x'],
            y=freethrow2['y'],
            z=np.zeros(len(freethrow2)),  # Place court lines on the floor
            mode='lines',
            line=dict(color='white', width=6),
            name="Court Perimeter",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=freethrow3['x'],
            y=freethrow3['y'],
            z=np.zeros(len(freethrow3)),  # Place court lines on the floor
            mode='lines',
            line=dict(color='white', width=6),
            name="Court Perimeter",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=freethrow4['x'],
            y=freethrow4['y'],
            z=np.zeros(len(freethrow4)),  # Place court lines on the floor
            mode='lines',
            line=dict(color='white', width=6),
            name="Court Perimeter",
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter3d(
            x=freethrow5['x'],
            y=freethrow5['y'],
            z=np.zeros(len(freethrow5)),  # Place court lines on the floor
            mode='lines',
            line=dict(color='white', width=6),
            name="Court Perimeter",
            hoverinfo='none'
        ))
        court_perimeter_bounds = np.array([[-250, 0, -0.2], [250, 0, -0.2], [250, 450, -0.2], [-250, 450, -0.2], [-250, 0, -0.2]])
        
        # Extract x, y, and z values for the mesh
        court_x = court_perimeter_bounds[:, 0]
        court_y = court_perimeter_bounds[:, 1]
        court_z = court_perimeter_bounds[:, 2]
        
        # Add a square mesh to represent the court floor at z=0
        fig.add_trace(go.Mesh3d(
            x=court_x,
            y=court_y,
            z=court_z,
            color='black',
            # opacity=0.5,
            name='Court Floor',
            hoverinfo='none',
            showscale=False
        ))
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            scene_aspectmode="data",
            height=600,
            scene_camera=dict(
                eye=dict(x=1.3, y=0, z=0.7)
            ),
            title="",
            scene=dict(
                xaxis=dict(title='', showticklabels=False, showgrid=False),
                    yaxis=dict(title='', showticklabels=False, showgrid=False),
                    zaxis=dict(title='',  showticklabels=False, showgrid=False,showbackground=False,backgroundcolor='black'),
        
        ),
        showlegend=False
        )
        with col2:
            st.plotly_chart(fig,use_container_width=True)
    else:
         st.error('Player Not Found')
