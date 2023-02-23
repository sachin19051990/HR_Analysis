import streamlit as st
import pandas as pd
import pickle
#from st_aggrid import AgGrid

model = pickle.load(open('model.pkl', 'rb'))


def predict(data):
    prediction = model.predict(data)
    return prediction


def main():
    st.title("")

# HR analytics solution:

html_temp = """ <div style="background-color:green;padding:10px;margin-bottom:30px;">
    <h3 style="color:white;text-align:center;">HR Analytics: Employee Promotion Prediction</h3>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

# ed = st.selectbox(
#     'Education',
#     ('Master\'s & above', 'Bachelor\'s', 'Below Secondary'))
# de = st.selectbox(
#     'Department',
#     ('Sales & Marketing', 'Operations', 'Technology', 'Analytics', 'R&D', 'Procurement', 'Finance', 'HR', 'Legal'))
# ge = st.selectbox(
#     'Gender',
#     ('m', 'f'))
# re = st.selectbox(
#     'Recruitment',
#     ('sourcing','other','referred'))
# reg = st.selectbox(
#     'Region',
#     ('region_7','region_22','region_19'))
# nt = st.number_input("Number of trainings",step=1)
# pr = st.number_input("Previous year rating", step=1)
# ls = st.number_input("Length of service", step=1)
# aw = st.number_input("Awards won", step=1)
# ats = st.number_input("Average training score", step=1)
# age = st.number_input("Age", step=1)
# st.write('you: ',ed)

uploaded_file=st.file_uploader('Choose a CSV file')
result = ""
if st.button("Predict"):
    result = predict(pd.read_csv(uploaded_file))
    # ans = 'No' if result == 0 else 'Yes'
    # st.success('The output is {}'.format(result))
    # st.success('The employee can be promoted: {}'.format(ans))
    st.text("1 if employee getting promoted otherwise 0")
    st.write(result)

if st.button("About"):
    st.text("Model used: ")
    st.text("Accuracy of the model: ")

main()

