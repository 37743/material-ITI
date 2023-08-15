import numpy as np
import pickle
import streamlit as st


#loading the model
loaded_model = pickle.load(open('/Users/ayaabdelsamad/Desktop/ML model as api/trained_model.sav', 'rb'))


#prediction function
def claim_prediction(input_data):
    input_data_as_array = np.asarray(input_data)
    input_data_reshape = input_data_as_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshape)
    
    if prediction[0] == 1:
        return 'Claim accepted'
    else:
        return 'Claim rejected'


#api
def main():
    
    #title of web page
    st.title('Insurance Claim Prediction')
    
    #taking input from user
    Income = st.text_input('Yearly income (Numeric Value)')
    application_underwriting_score = st.text_input('Application underwriting score (Numeric Value)')
    no_of_premiums_paid = st.text_input('Number of premiums paid (Numeric Value)')
    sourcing_channel = st.text_input('Sourcing channel (Numeric Value)')
    residence_area_type = st.text_input('Residence area type (Numeric Value)')
    age = st.text_input('User age (Numeric Value)')
    late_premium = st.text_input('Late premium (Numeric Value)')
    
    #output prediction
    output = ''
    
    #button for prediction 
    if st.button('Predict claim'):
        
        #check if input is numeric value
        ##
        
        output = claim_prediction([Income, application_underwriting_score, no_of_premiums_paid, sourcing_channel, residence_area_type, age, late_premium])
    
    #print output
    st.success(output)


if __name__ == '__main__':
    main()
    
    

