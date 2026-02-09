import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class CKDPredictionApp(App):
    def build(self):
        self.title = "CKD Prediction"
        
        # Load model
        try:
            self.model = joblib.load('pone_disease_prediction_model.pkl')
            self.model_loaded = True
        except:
            self.model_loaded = False
        
        # Main layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Title
        title = Label(
            text='🏥 CKD Risk Predictor',
            font_size='20sp',
            size_hint_y=None,
            height=50,
            bold=True
        )
        main_layout.add_widget(title)
        
        # Scrollable content
        scroll = ScrollView()
        content_layout = BoxLayout(orientation='vertical', size_hint_y=None, spacing=5)
        content_layout.bind(minimum_height=content_layout.setter('height'))
        
        # Input fields
        self.inputs = {}
        
        # Demographics
        content_layout.add_widget(Label(text='--- Demographics ---', font_size='16sp', bold=True))
        
        # Age
        age_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        age_layout.add_widget(Label(text='Age:', size_hint_x=0.3))
        self.inputs['age'] = TextInput(input_type='number', size_hint_x=0.7)
        age_layout.add_widget(self.inputs['age'])
        content_layout.add_widget(age_layout)
        
        # Gender
        gender_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        gender_layout.add_widget(Label(text='Gender:', size_hint_x=0.3))
        self.inputs['gender'] = Spinner(
            text='Female',
            values=('Female', 'Male'),
            size_hint_x=0.7
        )
        gender_layout.add_widget(self.inputs['gender'])
        content_layout.add_widget(gender_layout)
        
        # Medical History
        content_layout.add_widget(Label(text='--- Medical History ---', font_size='16sp', bold=True))
        
        # Diabetes
        diabetes_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        diabetes_layout.add_widget(Label(text='Diabetes:', size_hint_x=0.3))
        self.inputs['diabetes'] = Spinner(
            text='No',
            values=('No', 'Yes'),
            size_hint_x=0.7
        )
        diabetes_layout.add_widget(self.inputs['diabetes'])
        content_layout.add_widget(diabetes_layout)
        
        # Hypertension
        htn_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        htn_layout.add_widget(Label(text='Hypertension:', size_hint_x=0.3))
        self.inputs['htn'] = Spinner(
            text='No',
            values=('No', 'Yes'),
            size_hint_x=0.7
        )
        htn_layout.add_widget(self.inputs['htn'])
        content_layout.add_widget(htn_layout)
        
        # Clinical Measurements
        content_layout.add_widget(Label(text='--- Clinical Measurements ---', font_size='16sp', bold=True))
        
        # eGFR
        egfr_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        egfr_layout.add_widget(Label(text='eGFR:', size_hint_x=0.3))
        self.inputs['egfr'] = TextInput(input_type='number', size_hint_x=0.7)
        egfr_layout.add_widget(self.inputs['egfr'])
        content_layout.add_widget(egfr_layout)
        
        # Creatinine
        creat_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        creat_layout.add_widget(Label(text='Creatinine:', size_hint_x=0.3))
        self.inputs['creatinine'] = TextInput(input_type='number', size_hint_x=0.7)
        creat_layout.add_widget(self.inputs['creatinine'])
        content_layout.add_widget(creat_layout)
        
        # HbA1c
        hba1c_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        hba1c_layout.add_widget(Label(text='HbA1c:', size_hint_x=0.3))
        self.inputs['hba1c'] = TextInput(input_type='number', size_hint_x=0.7)
        hba1c_layout.add_widget(self.inputs['hba1c'])
        content_layout.add_widget(hba1c_layout)
        
        # BMI
        bmi_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        bmi_layout.add_widget(Label(text='BMI:', size_hint_x=0.3))
        self.inputs['bmi'] = TextInput(input_type='number', size_hint_x=0.7)
        bmi_layout.add_widget(self.inputs['bmi'])
        content_layout.add_widget(bmi_layout)
        
        # Predict button
        predict_btn = Button(
            text='🔮 Predict CKD Risk',
            size_hint_y=None,
            height=50,
            background_color=(0.2, 0.6, 1, 1),
            bold=True
        )
        predict_btn.bind(on_press=self.predict_ckd)
        content_layout.add_widget(predict_btn)
        
        # Result label
        self.result_label = Label(
            text='Enter patient data and click Predict',
            font_size='14sp',
            text_size=(400, None),
            halign='center'
        )
        content_layout.add_widget(self.result_label)
        
        scroll.add_widget(content_layout)
        main_layout.add_widget(scroll)
        
        return main_layout
    
    def predict_ckd(self, instance):
        if not self.model_loaded:
            self.show_popup('Error', 'Model not loaded. Please check the model file.')
            return
        
        try:
            # Collect input data
            data = {
                'StudyID': 1,
                'Gender': 0 if self.inputs['gender'].text == 'Female' else 1,
                'AgeBaseline': int(self.inputs['age'].text) if self.inputs['age'].text else 50,
                'Age.3.categories': int(self.inputs['age'].text) // 10 if self.inputs['age'].text else 5,
                'HistoryDiabetes': 1 if self.inputs['diabetes'].text == 'Yes' else 0,
                'HistoryCHD': 0,
                'HistoryVascular': 0,
                'HistorySmoking': 0,
                'HistoryHTN ': 1 if self.inputs['htn'].text == 'Yes' else 0,
                'HistoryDLD': 0,
                'HistoryObesity': 0,
                'DLDmeds': 0,
                'DMmeds': 1 if self.inputs['diabetes'].text == 'Yes' else 0,
                'HTNmeds': 1 if self.inputs['htn'].text == 'Yes' else 0,
                'ACEIARB': 0,
                'CholesterolBaseline': 200,
                'TriglyceridesBaseline': 1.5,
                'HgbA1C': float(self.inputs['hba1c'].text) if self.inputs['hba1c'].text else 6.0,
                'CreatnineBaseline': float(self.inputs['creatinine'].text) if self.inputs['creatinine'].text else 80,
                'eGFRBaseline': float(self.inputs['egfr'].text) if self.inputs['egfr'].text else 90,
                'sBPBaseline': 120,
                'dBPBaseline': 80,
                'BMIBaseline': float(self.inputs['bmi'].text) if self.inputs['bmi'].text else 25,
                'TimeToEventMonths': 35
            }
            
            # Convert to DataFrame
            import pandas as pd
            input_df = pd.DataFrame([data])
            
            # Make prediction
            prediction = self.model.predict(input_df)[0]
            probability = self.model.predict_proba(input_df)[0]
            
            # Display result
            prob_ckd = probability[1] * 100
            
            if prediction == 1:
                result_text = f"🔴 HIGH RISK: {prob_ckd:.1f}% CKD probability"
                color = (1, 0, 0, 1)  # Red
            else:
                result_text = f"🟢 LOW RISK: {prob_ckd:.1f}% CKD probability"
                color = (0, 1, 0, 1)  # Green
            
            self.result_label.text = result_text
            self.result_label.color = color
            
            # Show detailed popup
            self.show_detailed_result(prob_ckd, prediction)
            
        except Exception as e:
            self.show_popup('Error', f'Prediction failed: {str(e)}')
    
    def show_popup(self, title, message):
        popup = Popup(
            title=title,
            content=Label(text=message),
            size_hint=(0.8, 0.4)
        )
        popup.open()
    
    def show_detailed_result(self, prob_ckd, prediction):
        if prob_ckd < 20:
            risk_level = "Very Low"
            recommendation = "Regular monitoring recommended"
        elif prob_ckd < 40:
            risk_level = "Low"
            recommendation = "Annual check-up advised"
        elif prob_ckd < 60:
            risk_level = "Moderate"
            recommendation = "Consult nephrologist"
        elif prob_ckd < 80:
            risk_level = "High"
            recommendation = "Immediate medical attention"
        else:
            risk_level = "Very High"
            recommendation = "Urgent intervention required"
        
        message = f"""
Risk Level: {risk_level}
CKD Probability: {prob_ckd:.1f}%

Recommendation: {recommendation}

⚠️ This is for educational purposes only.
Consult healthcare professionals for medical decisions.
        """
        
        popup = Popup(
            title='🎯 Prediction Result',
            content=Label(text=message),
            size_hint=(0.9, 0.6)
        )
        popup.open()

if __name__ == '__main__':
    CKDPredictionApp().run()
