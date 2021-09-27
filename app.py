from flask import Flask
from flask import render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['TEMPLATES_AUTO_RELOAD']=True

#NBA LINEAR REGRESSION
stats = pd.read_csv('2021RegularStats.csv')
stats = stats.loc[stats['MIN'] > 5]
x = np.array(stats[['FG%', 'MIN']])
y = np.array(stats['PTS'])
model = linear_model.LinearRegression().fit(x, y)

#TITANIC TREE CLASSIFIER
titanic = pd.read_csv('titanic.csv')
#Data cleaning
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
#turn string input into numeric input
x_titanic = titanic[features]
x_titanic['Sex'] = np.where(x_train['Sex'] == 'male', 1, 0)
y_titanic = titanic.Survived

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/nba')
def calculate_points():
    minutes = request.args.get('minutes')
    fg = request.args.get('fg%')
    points = model.predict(np.array([fg, minutes]).reshape(1, -1).astype(np.float64))
    points=round(points[0], 1)
    if(points < 0):
        points = 0
    return render_template('nba.html', minutes=minutes, fg=fg, points=points)

@app.route('/titanic')
def calculate_survival():
