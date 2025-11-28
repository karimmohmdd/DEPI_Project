import sys
import os
#sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))



import streamlit as st
import pandas as pd
from src.predict import Predictor

st.set_page_config(
    page_title="Mental Disease & Health Risk",
    layout="wide"
)

st.title("Mental Health Risk Predictor")
st.markdown("### Please fill in the details below to assess the risk factor.")

@st.cache_resource
def get_predictor():
    # بنجيب مكان ملف app.py الحالي
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # بنركب المسار عليه عشان نوصل للموديل بدقة
    model_path = os.path.join(current_dir, "model", "pipeline.pkl")
    
    return Predictor(model_path=model_path)
try:
    predictor = get_predictor() 
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

with st.form("risk_assessment_form"):
    
    # --- Section 1: Demographics ---
    st.subheader("👤 Personal Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sex = st.selectbox("Sex", ["Male", "Female"])
        age = st.selectbox("Age Category", [
            'Young Adults (18-34)', 'Early Middle Age (35-44)', 
            'Late Middle Age (45-54)', 'Seniors (55-69)', 'Elderly (70+)'
        ])
    
    with col2:
        race = st.selectbox("Race / Ethnicity", [
            'White only, Non-Hispanic', 'Black only, Non-Hispanic', 
            'Hispanic', 'Other', 'Multiracial, Non-Hispanic'
        ])
        height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.75, step=0.01)
        
    with col3:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.1)

    st.divider()

    # --- Section 2: General Health ---
    st.subheader("🩺 General Health Condition")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        gen_health = st.selectbox("General Health Assessment", ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'], index=2)
        diabetes = st.selectbox("Had Diabetes?", ["No", "Yes"])
        
    with col5:
        phys_health = st.number_input("Physical Health (Days illness/month)", 0, 30, 0)
        ment_health = st.number_input("Mental Health (Days stress/month)", 0, 30, 0)
        
    with col6:
        sleep = st.number_input("Sleep Hours (Average)", 1, 24, 7)
        walking = st.selectbox("Difficulty Walking?", ["No", "Yes"])

    st.divider()

    # --- Section 3: Difficulties & Disabilities ---
    st.subheader("♿ Difficulties & Disabilities")
    col7, col8, col9 = st.columns(3)
    
    with col7:
        hearing = st.selectbox("Deaf or Hard of Hearing?", ["No", "Yes"])
        vision = st.selectbox("Blind or Vision Difficulty?", ["No", "Yes"])
    
    with col8:
        concentrating = st.selectbox("Difficulty Concentrating?", ["No", "Yes"])
        errands = st.selectbox("Difficulty Doing Errands?", ["No", "Yes"])

    with col9:
        dressing = st.selectbox("Difficulty Dressing/Bathing?", ["No", "Yes"])

    st.divider()

    # --- Section 4: Habits ---
    st.subheader("🏃 Lifestyle & Habits")
    col10, col11 = st.columns(2)
    
    with col10:
        activity = st.selectbox("Physical Activities?", ["Yes", "No"]) 
    
    with col11:
        alcohol = st.selectbox("Alcohol Drinkers?", ["No", "Yes"])

    # زرار التوقع
    submitted = st.form_submit_button("🔍 Analyze Risk")


if submitted:
    
    input_data = {
        'Sex': sex,
        'AgeCategory': age,
        'RaceEthnicityCategory': race,
        'HeightInMeters': height,
        'WeightInKilograms': weight,
        'GeneralHealth': gen_health,
        'HadDiabetes': diabetes,
        'PhysicalHealthDays': float(phys_health),
        'MentalHealthDays': float(ment_health),
        'SleepHours': float(sleep),
        'DifficultyWalking': walking,
        'DeafOrHardOfHearing': hearing,
        'BlindOrVisionDifficulty': vision,
        'DifficultyConcentrating': concentrating,
        'DifficultyErrands': errands,
        'DifficultyDressingBathing': dressing,
        'PhysicalActivities': activity,
        'AlcoholDrinkers': alcohol
    }
    
    with st.spinner("Processing data..."):
        try:
            result = predictor.predict(input_data)
            
            st.markdown("### Result:")
            
            if result['prediction'] == 1:
                st.error(f"⚠️ **High Risk Detected**")
                st.warning("Based on the input, there are indicators of potential Mental health risks. Please consult a specialist.")
            else:
                st.success(f"✅ **Low Risk Detected**")
                st.balloons()
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
 