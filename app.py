# 이 파일만 mlenv 가상환경 쓸거임
# data analysis - in jupyter notebook
# train model - in vscode
# flask app - in vscode
from flask import Flask,render_template,request
import pickle 
import pandas as pd

with open('bike_model_ridge.pkl','rb') as f:  # pickled된(pretrained)모델을 사용하려면 이 모델이 쓴 패키지들이 설치가 되있어야 한다. 
    model=pickle.load(f) # pip install sklearn을 했는데 pip install scikit-learn이 맞는 거다..
    
app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def main():
    if request.method == 'GET':
        
        return render_template('main.html')
    if request.method =='POST':
        temperature=request.form['temperature']
        humidity=request.form['humidity']
        windspeed=request.form['windspeed']
        
        input_variables=pd.DataFrame([[temperature,humidity,windspeed]],
                                      columns=['temperature', 'humidity', 'windspeed'],
                                       dtype=float,index=['input'])
        prediction=model.predict(input_variables)[0]
        return render_template('main.html',original_input={'Temperature':temperature,
                                                     'Humidity':humidity,
                                                     'Windspeed':windspeed},
                                     result=prediction)
# https://stackoverflow.com/questions/29086398/sklearn-turning-off-warnings/32389270        