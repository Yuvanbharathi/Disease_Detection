from random import choice
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from annotated_text import annotated_text

st.title('Disease Detection')

def main():
    menu = ['Home',"Cardiac Disease", "Cancer","Diabetes"]
    choice=st.sidebar.selectbox('Menu',menu)
    if choice =='Cardiac Disease':
        st.write("""
            # Heart Disease Analysis

            This app predicts Whether a person has Heart Disease or not
            """)

        Disease_raw=pd.read_csv('heart_disease_data.csv')
        Disease_raw.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
            'exercise_induced_angina', 'old_peak', 'st_slope','target']
        x=Disease_raw.drop('target',axis=1)
        y=Disease_raw['target']

        st.sidebar.header('User Input Parameter')

        def user_input():
            age=st.sidebar.slider('Age',10,80,60)
            sex=st.sidebar.slider('Gender',0,1,1)
            chest_pain_type=st.sidebar.slider('Chest Pain Type',1,4,2)
            resting_blood_pressure=st.sidebar.slider('BP',80,180,170)
            cholesterol=st.sidebar.slider('Cholestrol',0,500,250)
            fasting_blood_sugar =st.sidebar.slider('Blood Sugar',0,1,0)
            rest_ecg =st.sidebar.slider('ECG',0,2,1)
            max_heart_rate_achieved =st.sidebar.slider('Heart Rate',80,180,100)
            exercise_induced_angina =st.sidebar.slider('Induced Angina',0,1)
            old_peak=st.sidebar.slider('Old Peak',0,3,1)
            st_slope =st.sidebar.slider('Slope',1,2,1)
            data={'age':age, 'sex':sex, 'chest_pain_type':chest_pain_type, 'resting_blood_pressure':resting_blood_pressure,
                    'cholesterol':cholesterol, 'fasting_blood_sugar':fasting_blood_sugar, 'rest_ecg':rest_ecg, 'max_heart_rate_achieved':max_heart_rate_achieved,
                    'exercise_induced_angina': exercise_induced_angina, 'old_peak': old_peak,'st_slope':st_slope
                }
            features=pd.DataFrame(data,index=[0])
            return features
        df=user_input()
        st.subheader('User Input Features')
        st.write('Change the slider to modify the value and obtain outputs')
        st.write(df)
        st.write('---')

        model=RandomForestClassifier()
        model.fit(x, y)
        prediction= model.predict(df)

        probability = model.predict_proba(df)
        st.subheader("Prediction")
        st.write(prediction)
            
        st.subheader('Probability')
        st.write(probability)

        st.write('0 - No Heart disease')
        st.write('1 - Heart Disease')
    elif choice=='Cancer':
        st.write("""
            # Cancer Disease Detection

            This webpage predicts Whether a person has Cancer Disease or not
            """)

        Disease_raw=pd.read_csv('cancer.csv')
        Disease_raw.columns = ['id','diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean',
                            'concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se',
                            'perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se',
                            'fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
                            'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst','Unnamed_32']
        x=Disease_raw.drop('id',axis=1)
        a=pd.get_dummies(Disease_raw["diagnosis"])
        x=pd.concat([Disease_raw,a],axis="columns")
        x=x.drop('Unnamed_32',axis=1)
        x=x.drop('id',axis=1)
        x=x.drop('diagnosis',axis=1)

        x=x.drop('compactness_worst',axis=1)
        x=x.drop('concavity_worst',axis=1)
        x=x.drop('area_mean',axis=1)
        x=x.drop('concave points_mean',axis=1)
        x=x.drop('area_worst',axis=1)
        x=x.drop('perimeter_worst',axis=1)
        x=x.drop('radius_worst',axis=1)
        x=x.drop('concavity_mean',axis=1)
        x=x.drop('smoothness_worst',axis=1)
        x=x.drop('perimeter_mean',axis=1)
        x=x.drop('perimeter_se',axis=1)
        x=x.drop('area_se',axis=1)
        x=x.drop('texture_worst',axis=1)
        x=x.drop('concave points_worst',axis=1)
        x=x.drop('fractal_dimension_worst',axis=1)
        x=x.drop('concavity_se',axis=1)

        y=x['M']

        x=x.drop('B',axis=1)
        x=x.drop('M',axis=1)

        st.sidebar.header('User Input Parameter')

        def user_input():
            radius_mean=st.sidebar.slider('radius_mean',7.0,28.0,10.0)
            texture_mean=st.sidebar.slider('texture_mean',10.0,40.0,20.0)
            smothness_mean=st.sidebar.slider('smothness_mean',0.0,float(0.2),float(0.1))
            compactness_mean=st.sidebar.slider('compactness_mean',0.0,0.5,0.25)
            symmetry_mean=st.sidebar.slider('symmetry_mean',0.1,0.3,0.1)
            fractal_dimension_mean =st.sidebar.slider('fractal_dimension_mean',0.05,0.1,0.75)
            radius_se =st.sidebar.slider('radius_se',0.0,3.0,1.0)
            texture_se=st.sidebar.slider('texture_se',0.0,5.0,2.0)
            smoothness_se =st.sidebar.slider('smoothness_se',0.0,0.04,0.02)
            compactness_se=st.sidebar.slider('compactness_se',0.0,0.2,0.1)
            concave_points_se =st.sidebar.slider('concave_points_se',0.0,0.05,0.025)
            symmetry_se =st.sidebar.slider('symmetry_se',0.0,0.08,0.02)
            fractal_dimension_se =st.sidebar.slider('fractal_dimension_se',0.0,0.03,0.015)
            symmetry_worst =st.sidebar.slider('symmetry_worst',0.1,0.7,0.5)
            data={'radius_mean':radius_mean, 'texture_mean':texture_mean, 'smothness_mean':smothness_mean, 'compactness_mean':compactness_mean,
                    'symmetry_mean':symmetry_mean, 'fractal_dimension_mean':fractal_dimension_mean, 'radius_se':radius_se, 'texture_se':texture_se,
                    'smoothness_se': smoothness_se, 'compactness_se': compactness_se,'concave_points_se':concave_points_se,'symmetry_se':symmetry_se,
                    'fractal_dimension_se': fractal_dimension_se, 'symmetry_worst': symmetry_worst
                }
            features=pd.DataFrame(data,index=[0])
            return features
        df=user_input()

        st.subheader('User Input Features')
        st.write('Change the slider to modify the value and obtain outputs')
        st.write(df)
        st.write('---')

        model=RandomForestClassifier()
        model.fit(x, y)
        prediction= model.predict(df)

        probability = model.predict_proba(df)
        st.subheader("Prediction")
        st.write(prediction)

        st.subheader('Probability')
        st.write(probability)

        st.write('0 - benign (noncancerous)')
        st.write('1 - malignant (cancerous)')
    elif choice=='Diabetes':
        st.write("""
            # Diabetes Disease Detection

            This webpage predicts Whether a person has Diabetes Disease or not
            """)

        Disease_raw=pd.read_csv('diabetes.csv')
        Disease_raw.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',"DiabetesPedigreeFunction",'Age','Outcome']
        x=Disease_raw.drop('Outcome',axis=1)
        y=Disease_raw['Outcome']

        st.sidebar.header('User Input Parameter')

        def user_input():
            Pregnancies=st.sidebar.slider('Pregnancies',0,20,3)
            Glucose=st.sidebar.slider('Glucose',100,200,120)
            BloodPressure=st.sidebar.slider('BloodPressure',40,130,70)
            SkinThickness=st.sidebar.slider('SkinThickness',0,100,20)
            Insulin=st.sidebar.slider('Insulin',0,846,80)
            BMI =st.sidebar.slider('BMI',0,70,30)
            DiabetesPedigreeFunction =st.sidebar.slider('DiabetesPedigreeFunction',0.0,2.5,1.2)
            Age =st.sidebar.slider('age',20,80,50)
            data={'Age':Age, 'DiabetesPedigreeFunction':DiabetesPedigreeFunction, 'Glucose':Glucose, 'Pregnancies':Pregnancies,
                    'BloodPressure':BloodPressure, 'SkinThickness':SkinThickness, 'Insulin':Insulin, 'BMI':BMI}
            features=pd.DataFrame(data,index=[0])
            return features
        df=user_input()
        st.subheader('User Input Features')
        st.write('Change the slider to modify the value and obtain outputs')
        st.write(df)
        st.write('---')
   
        xm = RandomForestClassifier()
        xm.fit(x,y)
        prediction=xm.predict(df)
        probability =xm.predict_proba(df)
        st.subheader("Prediction")
        st.write(prediction)

        st.subheader('Probability')
        st.write(probability)

        st.write('0 - No Diabetes')
        st.write('1 - Diabetes')
        st.write(df)
        
    
        
        
        
    elif choice=='Home':
        st.subheader('This webpage will concentrate on detecting the person with various toxic diseases on their body')
        ##annotated_text(('This webpage will concentrate on detecting the person with various toxic diseases on their body','#8ef'),)
        st.radio(label= 'The different types of diseases given in this website are', options= ['Heart disease', 'Malaria'])
if __name__ == '__main__':
    	main()
