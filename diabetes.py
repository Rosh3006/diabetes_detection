from flask import Flask,render_template,request
import pickle

app=Flask(__name__)
model=pickle.load(open('D:\\flask\\diabetes\\saveModelDiabetes.sav','rb'))

@app.route('/')
def home():
    result=''
    return render_template('diabetes.html',**locals())

@app.route('/outcome',methods=['GET','POST'])
def outcome():
    if request.method=='POST':
        pregnancies=float(request.form['pregnancies'])
        glucose= float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_fun = float(request.form['diabetes_pedigree_fun'])
        age = float(request.form['age'])
        
        y = model.predict([[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree_fun,age]])[0]
        result=''
        if y==0:
            result="The person is free from diabetes"
        else:
            result="The person is suffering from diabetes"

    return render_template('diabetes.html',**locals())

if __name__== '__main__':
    app.run(debug=True)

