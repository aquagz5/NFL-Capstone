# NFL-Big-Data-Bowl

## Quick navigation
[Background](#background)  

[Data](#data)  

[Models](#models)  

[Timeline](#timeline)  

[Repo Structure](#repo-structure)  

[Logistics](#project-logistics)  

[Resources](#resources)  

[Contact](#contact-info) 

[Helpful Links](#helpful-links)

## Goal

American football can be a complex sport to understand and play. However, when an offensive player tries to advance the play as a ball carrier, the defense only has one job: tackle. The objective is to get the ball carrier down and as fast as possible before gaining more yards. As a ball carrier, your job is simple as well: move the ball down the field as far as possible, dodging tackles. 

With data, the NFL community has evolved its analytics allowing for data-driven strategies, detailed metrics on player performance, and advanced statistics for viewership. There has been many advanced analytics performed on aspects of the game such as blitz prediction, catch probability, expected passing yards, and win probability. However, there has been limited analytics on the aspect of tackling and the probability that a tackle will be made on a given play. In addition, simple statistics like total tackles, tackle assists, and missed tackles do not tell the full picture of a player’s performance with tacking. 

Our goal for this year’s NFL Big Data Bowl, is to provide a model with accurate probabilistic predictions that a defensive player will make a tackle on any given play from scrimmage. The model will encompass features that are key components to making a tackle in the game, such as player orientation, defense formation, distance from ball, and distance from potential ball carriers. With this advanced metric we hope to provide the following use cases and value propositions for the NFL:
An accurate tackling probability metric to advance real-time game analytics for viewership 
With an accurate tackling probability metric, we can also identify improvements in modern tackling statistics to encompass a more comprehensive understanding of player and team performance
In addition, we hope to provide teams with a dynamic tool that will allow them to develop in-game strategies based on an accurate tackling probability
 

## Background  

Provide a broad overview of the purpose of the project.

## Data

Describe the data - what kind of data is it?  Describe the general format, and potential quirks.

### Data security

If there are any security concerns or requirements regarding the data, they should be described here.

### Counts

Describe the overall size of the dataset and the relative ratio of positive/negative examples for each of the response variables.

## Models

Clearly identify each of the response variables of interest.  Any additional desired analysis should also be described here.

## Timeline

Outline the desired timeline of the project and any explicit deadlines.

## Repo Structure 

Give a description of how the repository is structured. Example structure description below:

The repo is structured as follows: Notebooks are grouped according to their series (e.g., 10, 20, 30, etc) which reflects the general task to be performed in those notebooks.  Start with the *0 notebook in the series and add other investigations relevant to the task in the series (e.g., `11-cleaned-scraped.ipynb`).  If your notebook is extremely long, make sure you've utilized nbdev reuse capabilities and consider whether you can divide the notebook into two notebooks.

All files which appear in the repo should be able to run, and not contain error or blank cell lines, even if they are relatively midway in development of the proposed task. All notebooks relating to the analysis should have a numerical prefix (e.g., 31-) followed by the exploration (e.g. 31-text-labeling). Any utility notebooks should not be numbered, but be named according to their purpose. All notebooks should have lowercase and hyphenated titles (e.g., 10-process-data not 10-Process-Data). All notebooks should adhere to literate programming practices (i.e., markdown writing to describe problems, assumptions, conclusions) and provide adequate although not superfluous code comments.

## Project logistics

**Sprint planning**:  
- We will meet as a group in-person every Monday at 7:30

**Data location**: 
- Data can be found on the NFL Big Data Bowl Kaggle website: https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/data


## Contact Info

**Grant Duncan**

Email: GrantDuncan98@gmail.com


**Logan King**

Email: kinglogana97@gmail.com


**Lauren Manis**

Email: laurenmanis2@gmail.com


**Tony Quagliata**

Email: aquagliata5@gmail.com

## Helpful Links

- Kaggle competition Link: https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/overview
- Google Drive for NFL Big Data Project: https://drive.google.com/drive/folders/11aA_8YlZU7cMyx7HF7Kg66LRGgcyUjQp?usp=drive_link
- Example Notebook for tackling: https://www.kaggle.com/code/danitreisman/tackle-probability-and-the-value-of-a-tackle
- Prediction of Defensive Player Trajectories in NFL Games with Defender CNN-LSTM Model: https://assets.amazon.science/8f/31/de231564410aa55e346d02c34c12/prediction-of-defensive-player-trajectories-in-nfl-games-with-defender-cnn-lstm-model.pdf
