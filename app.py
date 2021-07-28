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

#SET UP MODEL
stats = pd.read_csv('2021RegularStats.csv')
stats = stats.loc[stats['MIN'] > 5]
x = np.array(stats[['FG%', 'MIN']])
y = np.array(stats['PTS'])

model = linear_model.LinearRegression().fit(x, y)


@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/calc')
def calculate_points():
    minutes = request.args.get('minutes')
    fg = request.args.get('fg%')
    points = model.predict(np.array([fg, minutes]).reshape(1, -1).astype(np.float64))
    points=round(points[0], 1)
    if(points < 0):
        points = 0
    return render_template('prediction.html', minutes=minutes, fg=fg, points=points)