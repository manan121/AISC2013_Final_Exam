from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Helper function to train model and get results
def train_model(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    return mse

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            mse = train_model(df)
            return render_template('result.html', mse=mse)
    
    return '''
    <!doctype html>
    <title>Upload your CSV file</title>
    <h1>Upload CSV file for analysis</h1>
    <form action="/" method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/manual', methods=['GET', 'POST'])
def manual_input():
    if request.method == 'POST':
        data = request.form['data']
        from io import StringIO
        df = pd.read_csv(StringIO(data))
        mse = train_model(df)
        return render_template('result.html', mse=mse)
    
    return '''
    <!doctype html>
    <title>Manual Data Entry</title>
    <h1>Enter data in CSV format</h1>
    <form action="/manual" method=post>
      <textarea name=data rows=10 cols=30></textarea>
      <input type=submit value=Submit>
    </form>
    '''

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == "__main__":
    app.run(debug=True)
