from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import os
from predictor import predict_yield, evaluate_surplus, build_input_df

app = Flask(__name__)
# Load population data once
pop_df = pd.read_csv(os.path.join('data','processed','population_processed.csv'))

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        area = request.form['area']
        year = int(request.form['year'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        fertilizer = float(request.form['fertilizer'])
        demand_pp = float(request.form.get('demand_pp', 1.0))

        input_df = build_input_df(area, year, pop_df, rainfall, temperature, fertilizer)
        pred = predict_yield(input_df)
        population = int(input_df['Population'].values[0])
        eval_res = evaluate_surplus(pred, population, demand_pp)

        return render_template('result.html',
                               area=area,
                               year=year,
                               rainfall=rainfall,
                               temperature=temperature,
                               fertilizer=fertilizer,
                               predicted_yield=pred,
                               population=population,
                               status=eval_res['status'],
                               amount=eval_res['amount'],
                               advice=eval_res['advice'])
    # GET request
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
