# Databricks notebook source
# MAGIC %md
# MAGIC # Model Evaluation Functions
# MAGIC
# MAGIC The following functions perform the neccessary tasks to evaluate model performance. The last function includes a function that visualizes the data along with the tackle probabilities. We used the following notebook as a base visualization and added our own nuances: https://www.kaggle.com/code/huntingdata11/animated-and-interactive-nfl-plays-in-plotly.

# COMMAND ----------

def log_loss(y_true, y_pred):
    import numpy as np
    epsilon = 1e-15  # to prevent log(0) which is undefined
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clip predicted probabilities to prevent log(0) or log(1)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# COMMAND ----------

def soft_max(df):
    def softmax(predictions):
        sum_preds = sum(predictions)
        if sum_preds == 0:
            normalized_preds = [0 for p in predictions]
        else:
            normalized_preds = [p/sum_preds for p in predictions]
        return normalized_preds
    df = df.dropna(subset = ['probs'])
    df['norm'] = df.groupby(['gameId', 'playId', 'frameId'])['probs'].transform(softmax)
    df = df[["gameId", "playId", "frameId", "nflId", "norm"]]

    return df


# COMMAND ----------

#plotROC
# Purpose: the following fuction will plot the ROC curve for a model
# Input: actual- array of 1 and 0 of the testing data
#        predicted_prob - array of predicted probabilities based on model
#        title_model - insert string of the model name for title
# Output: figure of ROC curve
def plotROC(actual, predicted_prob, title_model = ''):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    if title_model == '':
        title_plot = 'ROC'
    else:
        title_plot = 'ROC: ' + title_model
    
    fpr, tpr, thresholds = roc_curve(actual, predicted_prob)
    model_roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.subplots(1, figsize=(7,7))
    plt.title(title_plot, fontsize = 14, weight = 'bold')
    plt.plot(fpr,tpr)
    plt.plot([0,1], ls = "--")
    plt.plot([0,0], [1,0], c=".7"), plt.plot([1,1], c = ".7")
    plt.ylabel('True Positive Rate', fontsize = 10)
    plt.xlabel('False Positive Rate', fontsize = 10)
    plt.text(0.55, 0.00,
            "ROC curve (area = %0.5f)" % model_roc_auc, fontsize = 12,
            bbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5))
    plt.show()
    return

# COMMAND ----------

#youdens_j 
# Purpose: statistic to return optimal threshold 
# Input: actual - array of 0s and 1s based on actual testing observations
#        predicted_prob - probabilities of predictions from model
# Output: returns a value between 0 and 1 that will act as the optimal threshold value
def youdens_j(actual, predicted_prob):
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(actual, predicted_prob)
    idx = np.argmax(tpr-fpr)
    threshold_val = thresholds[idx]
    
    return(threshold_val)

# COMMAND ----------

#binaryClfMetrics
# Purpose: The purpose of this function is to provide important metrics based on confusion matrix of threshold as well as 
#          plot the confusion matrix
# Input: actual - array of 1 and 0 of actual test result
#        predicted_prob - array of predicted probabilities from model
#        threshold - cutoff point to find predicted 0 or 1
# Output: List of metrix, plot of confusion matrix report, plot of confusion matrix
def binaryClfMetrics(actual, predicted_prob,threshold):
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    
    preds = [1 if x > threshold else 0 for x in predicted_prob]
    
    print('Binary Classifier Metrics')
    print('\tROC AUC score: \t\t%0.5f' % roc_auc_score(y_true = actual, y_score = predicted_prob))
    print('\tAccuracy:\t\t%0.5f' % accuracy_score(y_true = actual, y_pred = preds))
    print('\tFalse-Postiive rate: \t%0.5f' % (1-precision_score(y_true=actual, y_pred = preds, labels = None, pos_label = 1,
                                                               average = 'weighted', sample_weight = None)))
    print('\tF1 score (weighted):\t%0.5f' % f1_score(y_true = actual, y_pred = preds, labels = None, pos_label = 1, 
                                                    average = 'weighted', sample_weight = None))
    print('Classification Report:')
    print(classification_report(y_true = actual, y_pred = preds))
    cm = confusion_matrix(actual, preds)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    return

# COMMAND ----------

def highest_avg_acc_per_play(data,probabilities):
    import pandas as pd

    data["probs"] = pd.Series(probabilities)

    play_tacklers = data.drop_duplicates(subset = ['gamePlayId','nflId','tackle_multiple'])[['gamePlayId','nflId','tackle_multiple']]
    play_tacklers = play_tacklers[play_tacklers["tackle_multiple"]==1][["gamePlayId", "nflId"]]

    ids_max = data.groupby(["gamePlayId", "frameId"])['probs'].idxmax()
    frame_max_prob = data.loc[ids_max][["gamePlayId", "nflId"]]
    occurences_per = frame_max_prob.groupby(['gamePlayId', 'nflId']).size() / frame_max_prob.groupby('gamePlayId').size() * 100
    max_occurrence_idx = occurences_per.groupby('gamePlayId').idxmax()
    highest_pred = occurences_per.loc[max_occurrence_idx].reset_index()[["gamePlayId", "nflId"]]

    checks = pd.merge(play_tacklers, highest_pred, on = "gamePlayId", how = "inner")
    checks['is_correct'] = (checks['nflId_x']==checks['nflId_y']).astype(int)
    checks_play = checks.groupby('gamePlayId')['is_correct'].max()
    return(sum(checks_play)/len(checks_play))


# COMMAND ----------

def acc_frame_tackle(data,probabilities):
    import pandas as pd
    data["probabilities"] = pd.Series(probabilities)

    frame_ids = data[data["tackle_single"]==1].drop_duplicates(subset = ["gamePlayId","frameId"])[["gamePlayId", "frameId"]]
    
    tackle_events = data.merge(frame_ids, on = ["gamePlayId", "frameId"], how = "inner")[["gamePlayId", "nflId", "frameId","tackle_single", "probabilities"]]

    grouped_data = tackle_events.groupby(['gamePlayId', 'frameId'])
    max_prob_data = grouped_data.apply(lambda x: x.loc[x['probabilities'].idxmax()])[["gamePlayId","nflId"]]
    max_prob_data["Highest"] = 1
    max_prob_data.reset_index(drop=True, inplace=True)

    tackle_events = tackle_events.merge(max_prob_data, on = ["gamePlayId","nflId"], how = "left")
    tackle_events["Highest"].fillna(0, inplace=True)
    tackle_events['predicted_tackler'] = (tackle_events['tackle_single'] == tackle_events["Highest"])
    tackle_events = tackle_events[tackle_events["tackle_single"]==1]
    grouped_data = tackle_events.groupby('gamePlayId')['predicted_tackler'].any()
    grouped_data = grouped_data.reset_index()
    accuracy = grouped_data['predicted_tackler'].mean()
    return accuracy

# COMMAND ----------

def animate_play_probs(games,tracking_df,play_df,players,model_data, probs,gameId, playId):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    colors = {
        'ARI':["#97233F","#000000","#FFB612"], 
        'ATL':["#A71930","#000000","#A5ACAF"], 
        'BAL':["#241773","#000000"], 
        'BUF':["#00338D","#C60C30"], 
        'CAR':["#0085CA","#101820","#BFC0BF"], 
        'CHI':["#0B162A","#C83803"], 
        'CIN':["#FB4F14","#000000"], 
        'CLE':["#311D00","#FF3C00"], 
        'DAL':["#003594","#041E42","#869397"],
        'DEN':["#FB4F14","#002244"], 
        'DET':["#0076B6","#B0B7BC","#000000"], 
        'GB' :["#203731","#FFB612"], 
        'HOU':["#03202F","#A71930"], 
        'IND':["#002C5F","#A2AAAD"], 
        'JAX':["#101820","#D7A22A","#9F792C"], 
        'KC' :["#E31837","#FFB81C"], 
        'LA' :["#003594","#FFA300","#FF8200"], 
        'LAC':["#0080C6","#FFC20E","#FFFFFF"], 
        'LV' :["#000000","#A5ACAF"],
        'MIA':["#008E97","#FC4C02","#005778"], 
        'MIN':["#4F2683","#FFC62F"], 
        'NE' :["#002244","#C60C30","#B0B7BC"], 
        'NO' :["#101820","#D3BC8D"], 
        'NYG':["#0B2265","#A71930","#A5ACAF"], 
        'NYJ':["#125740","#000000","#FFFFFF"], 
        'PHI':["#004C54","#A5ACAF","#ACC0C6"], 
        'PIT':["#FFB612","#101820"], 
        'SEA':["#002244","#69BE28","#A5ACAF"], 
        'SF' :["#AA0000","#B3995D"],
        'TB' :["#D50A0A","#FF7900","#0A0A08"], 
        'TEN':["#0C2340","#4B92DB","#C8102E"], 
        'WAS':["#5A1414","#FFB612"], 
        'football':["#CBB67C","#663831"]
    }

    def hex_to_rgb_array(hex_color):
        '''take in hex val and return rgb np array'''
        return np.array(tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))) 

    def ColorDistance(hex1,hex2):
        '''d = {} distance between two colors(3)'''
        if hex1 == hex2:
            return 0
        rgb1 = hex_to_rgb_array(hex1)
        rgb2 = hex_to_rgb_array(hex2)
        rm = 0.5*(rgb1[0]+rgb2[0])
        d = abs(sum((2+rm,4,3-rm)*(rgb1-rgb2)**2))**0.5
        return d

    def ColorPairs(team1,team2):
        color_array_1 = colors[team1]
        color_array_2 = colors[team2]
        # If color distance is small enough then flip color order
        if ColorDistance(color_array_1[0],color_array_2[0])<500:
            return {team1:[color_array_1[0],color_array_1[1]],team2:[color_array_2[1],color_array_2[0]],'football':colors['football']}
        else:
            return {team1:[color_array_1[0],color_array_1[1]],team2:[color_array_2[0],color_array_2[1]],'football':colors['football']}
        
    
    tracking_play = tracking_df[(tracking_df["gameId"]==gameId)&(tracking_df["playId"]==playId)]
    tracking_players_df = pd.merge(tracking_play,players,how="left",on = "nflId")
    tracking_players_df = pd.merge(tracking_players_df,play_df,how="left",on = ["gameId","playId"])
    tracking_players_df = pd.merge(tracking_players_df,games,how="left", on = "gameId")
    model_data["probs"] = pd.Series(probs)

    tracking_players_df = tracking_players_df.merge(model_data[['gameId','playId','nflId','frameId','tackle_multiple', 'probs']], on=['gameId','playId','nflId','frameId'], how='left')

    last_frame = tracking_players_df.loc[~tracking_players_df['probs'].isna(), 'frameId'].max()
    tracking_players_df = tracking_players_df[tracking_players_df['frameId'] <= last_frame-2]

    tracking_players_df['probs'] = tracking_players_df['probs'].fillna(0)
    tracking_players_df = tracking_players_df.merge(
        soft_max(tracking_players_df), on = ["gameId", "playId", "frameId", "nflId"], how = "inner")
    display(tracking_players_df)
    #display(tracking_players_df)
    #print(tracking_players_df.dtypes)

    sorted_frame_list = tracking_players_df.frameId.unique()
    sorted_frame_list.sort()
    
    # get good color combos
    team_combos = list(set(tracking_players_df.club.unique())-set(["football"]))
    color_orders = ColorPairs(team_combos[0],team_combos[1])
    
    # get play General information 
    line_of_scrimmage = tracking_players_df.absoluteYardlineNumber.values[0]
    ## Fixing first down marker issue from last year
    if tracking_players_df.playDirection.values[0] == "right":
        first_down_marker = line_of_scrimmage + tracking_players_df.yardsToGo.values[0]
    else:
        first_down_marker = line_of_scrimmage - tracking_players_df.yardsToGo.values[0]
    down = tracking_players_df.down.values[0]
    quarter = tracking_players_df.quarter.values[0]
    gameClock = tracking_players_df.gameClock.values[0]
    playDescription = tracking_players_df.playDescription.values[0]
    # Handle case where we have a really long Play Description and want to split it into two lines
    if len(playDescription.split(" "))>15 and len(playDescription)>115:
        playDescription = " ".join(playDescription.split(" ")[0:16]) + "<br>" + " ".join(playDescription.split(" ")[16:])

    # initialize plotly start and stop buttons for animation
    updatemenus_dict = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]
    # initialize plotly slider to show frame position in animation
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Frame:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }


    frames = []
    for frameId in sorted_frame_list:
        data = []
        # Add Numbers to Field 
        data.append(
            go.Scatter(
                x=np.arange(20,110,10), 
                y=[5]*len(np.arange(20,110,10)),
                mode='text',
                text=list(map(str,list(np.arange(20, 61, 10)-10)+list(np.arange(40, 9, -10)))),
                textfont_size = 30,
                textfont_family = "Courier New, monospace",
                textfont_color = "#ffffff",
                showlegend=False,
                hoverinfo='none'
            )
        )
        data.append(
            go.Scatter(
                x=np.arange(20,110,10), 
                y=[53.5-5]*len(np.arange(20,110,10)),
                mode='text',
                text=list(map(str,list(np.arange(20, 61, 10)-10)+list(np.arange(40, 9, -10)))),
                textfont_size = 30,
                textfont_family = "Courier New, monospace",
                textfont_color = "#ffffff",
                showlegend=False,
                hoverinfo='none'
            )
        )
        # Add line of scrimage 
        data.append(
            go.Scatter(
                x=[line_of_scrimmage,line_of_scrimmage], 
                y=[0,53.5],
                line_dash='dash',
                line_color='blue',
                showlegend=False,
                hoverinfo='none'
            )
        )
        # Add First down line 
        data.append(
            go.Scatter(
                x=[first_down_marker,first_down_marker], 
                y=[0,53.5],
                line_dash='dash',
                line_color='yellow',
                showlegend=False,
                hoverinfo='none'
            )
        )
        # Add Endzone Colors 
        endzoneColors = {0:color_orders[tracking_players_df.homeTeamAbbr.values[0]][0],
                         110:color_orders[tracking_players_df.visitorTeamAbbr.values[0]][0]}
        for x_min in [0,110]:
            data.append(
                go.Scatter(
                    x=[x_min,x_min,x_min+10,x_min+10,x_min],
                    y=[0,53.5,53.5,0,0],
                    fill="toself",
                    fillcolor=endzoneColors[x_min],
                    mode="lines",
                    line=dict(
                        color="white",
                        width=3
                        ),
                    opacity=1,
                    showlegend= False,
                    hoverinfo ="skip"
                )
            )
        def_team = tracking_players_df["defensiveTeam"][0]
        tackle_ids = tracking_players_df[tracking_players_df["tackle_multiple"] == 1]["nflId"].unique()

        # Plot Players
        for team in tracking_players_df.club.unique():
            plot_df = tracking_players_df[(tracking_players_df.club == team) & (tracking_players_df.frameId == frameId)].copy()
            plot_df['marker_line_color'] = 'black'
            plot_df.loc[~plot_df['nflId'].isin(tackle_ids), 'marker_line_color'] = 'rgba(0, 0, 0, 0)'  # Set line color to None for players not in tackle_ids
            if team != "football" and team == def_team:
                scatter_trace = go.Scatter(
                    x=plot_df["x"],
                    y=plot_df["y"],
                    mode='markers',
                    marker=dict(
                        color=plot_df["norm"],
                        colorscale='YlOrRd',
                        cmin = 0,
                        cmax=1,
                        size=10,  # Adjust the size based on tackle_probability
                        opacity=0.9,  # Set alpha to 0.5 (transparency)
                        showscale=True,
                        symbol = "x",
                        colorbar = dict(
                            title = "Tackle Probability"
                        ),
                        line = dict(
                            color = plot_df["marker_line_color"],
                            width = 2
                        )
                    ),
                    hovertext=["nflId:{}<br>Tackle Probability: {:.10%}".format(nflId, norm) for nflId, norm in zip(plot_df["nflId"], plot_df["norm"])],
                    hoverinfo= "text",
                    showlegend=False
                )
                data.append(scatter_trace)
            elif team != "football":
                for _, player_row in plot_df.iterrows():
                    # Update size based on tackle_probability
                    data.append(go.Scatter(
                        x=[player_row["x"]],
                        y=[player_row["y"]],
                        mode='markers',
                        marker=dict(
                            color="black",
                            size=10,  # Adjust the size based on tackle_probability
                        ),
                        hoverinfo="none",
                        showlegend=False
                    ))
            else:
                data.append(go.Scatter(
                    x=plot_df["x"],
                    y=plot_df["y"],
                    mode='markers',
                    marker=dict(
                        color="brown",
                        line=go.scatter.marker.Line(width=2, color=color_orders[team][1]),
                        size=7,
                        opacity=0.9,  # Set alpha to 0.5 (transparency)
                        symbol = "diamond-wide"
                    ),
                    hoverinfo="none",
                    showlegend=False
                ))



        # add frame to slider
        slider_step = {"args": [
            [frameId],
            {"frame": {"duration": 100, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 0}}
        ],
            "label": str(frameId),
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)
        frames.append(go.Frame(data=data, name=str(frameId)))

    scale=10
    layout = go.Layout(
        autosize=False,
        width=120*scale,
        height=60*scale,
        xaxis=dict(range=[0, 120], autorange=False, tickmode='array',tickvals=np.arange(10, 111, 5).tolist(),showticklabels=False),
        yaxis=dict(range=[0, 53.3], autorange=False,showgrid=False,showticklabels=False),

        plot_bgcolor='#00B140',
        # Create title and add play description at the bottom of the chart for better visual appeal
        title=f"GameId: {gameId}, PlayId: {playId}<br>{gameClock} {quarter}Q"+"<br>"*19+f"{playDescription}",
        updatemenus=updatemenus_dict,
        sliders = [sliders_dict]
    )

    fig = go.Figure(
        data=frames[0]["data"],
        layout= layout,
        frames=frames[1:]
    )
    # Create First Down Markers 
    for y_val in [0,53]:
        fig.add_annotation(
                x=first_down_marker,
                y=y_val,
                text=str(down),
                showarrow=False,
                font=dict(
                    family="Courier New, monospace",
                    size=16,
                    color="black"
                    ),
                align="center",
                bordercolor="black",
                borderwidth=2,
                borderpad=4,
                bgcolor="#ff7f0e",
                opacity=1
                )
    # Add Team Abbreviations in EndZone's
    for x_min in [0,110]:
        if x_min == 0:
            angle = 270
            teamName=tracking_players_df.homeTeamAbbr.values[0]
        else:
            angle = 90
            teamName=tracking_players_df.visitorTeamAbbr.values[0]
        fig.add_annotation(
            x=x_min+5,
            y=53.5/2,
            text=teamName,
            showarrow=False,
            font=dict(
                family="Courier New, monospace",
                size=32,
                color="White"
                ),
            textangle = angle
        )
    return fig
