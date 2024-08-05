import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import streamlit as st
import joblib
import pandas as pd
import os

scaler_path = 'scaler.pkl'
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    st.error(f"Scaler file not found at path: {scaler_path}")


# Input fields for the user
age = st.number_input('Age', min_value=0, max_value=100)
sex = st.selectbox('Sex', ['Male', 'Female'])
highschool = st.selectbox('High School', ['Yes', 'No'])
scholarship = st.selectbox('Scholarship', ['Yes', 'No'])
work = st.selectbox('Work', ['Yes', 'No'])
extracurricular = st.selectbox('ExtraCurricular', ['Yes', 'No'])
relationship = st.selectbox('Relationship', ['Yes', 'No'])
salary = st.number_input('Salary', min_value=0)
transportation = st.selectbox('Transportation', ['Public', 'Private'])
accommodation = st.selectbox('Accommodation', ['On-campus', 'Off-campus'])
mother_edu = st.selectbox('Mother Education', ['Primary', 'Secondary', 'Higher'])
father_edu = st.selectbox('Father Education', ['Primary', 'Secondary', 'Higher'])
siblings = st.number_input('Siblings', min_value=0)
parental_stat = st.selectbox('Parental Status', ['Together', 'Separated'])
mother_occupation = st.text_input('Mother Occupation')
father_occupation = st.text_input('Father Occupation')
course_id = st.text_input('Course ID')

# When the user clicks the 'Predict' button
if st.button('Predict'):
    # Map categorical inputs to numerical values (if necessary)
    sex_map = {'Male': 1, 'Female': 0}
    highschool_map = {'Yes': 1, 'No': 0}
    scholarship_map = {'Yes': 1, 'No': 0}
    work_map = {'Yes': 1, 'No': 0}
    extracurricular_map = {'Yes': 1, 'No': 0}
    relationship_map = {'Yes': 1, 'No': 0}
    transportation_map = {'Public': 0, 'Private': 1}
    accommodation_map = {'On-campus': 0, 'Off-campus': 1}
    education_map = {'Primary': 0, 'Secondary': 1, 'Higher': 2}
    parental_stat_map = {'Together': 0, 'Separated': 1}

    # Create a dictionary from user inputs
    user_input = {
        'Age': age,
        'Sex': sex_map[sex],
        'HighSchool': highschool_map[highschool],
        'Scholarship': scholarship_map[scholarship],
        'Work': work_map[work],
        'ExtraCurricular': extracurricular_map[extracurricular],
        'Relationship': relationship_map[relationship],
        'Salary': salary,
        'Transportation': transportation_map[transportation],
        'Accommodation': accommodation_map[accommodation],
        'Mother Edu': education_map[mother_edu],
        'Father Edu': education_map[father_edu],
        'Siblings': siblings,
        'Parental Stat': parental_stat_map[parental_stat],
        'Mother Occupation': mother_occupation,  # Assuming no transformation needed
        'Father Occupation': father_occupation,  # Assuming no transformation needed
        'CourseID': course_id  # Assuming no transformation needed
    }

    # Convert the dictionary to a dataframe
    user_input_dataframe = pd.DataFrame([user_input])

    # Scale the input data
    user_input_scaled = scaler.transform(user_input_dataframe)

    # Make prediction
    prediction = model.predict(user_input_scaled)
    int_prediction = int(round(prediction[0], 0))

    # Display the result
    st.write(f'Predicted Performance: {int_prediction}')
