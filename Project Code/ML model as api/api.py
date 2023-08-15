import numpy as np
import pickle
import streamlit as st


#loading the model
loaded_model = pickle.load(open('ML model as api/trained_model.sav', 'rb'))


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
    Income = st.text_input('Yearly income (Numeric Value: 0:(0-23K) , 1:(23-109k), 2:(109-194k), 3:(194-279k), 4:(279-364k), 5:(364-450k), 6:(>4]50k))')
    application_underwriting_score = st.text_input('Application underwriting score (Numeric Value)')
    no_of_premiums_paid = st.text_input('Number of premiums paid (Numeric Value)')
    sourcing_channel = st.selectbox('Sourcing channel', ("--", "A", "B", "C", "D", "E"))
    residence_area_type = st.selectbox('Residence area type', ("--", "Rural", "Urban"))
    age = st.text_input('User age in days(Numeric Value:  0(:0-37), 1:(38-53), 2:(54-70), 3:(71-86), 4:(>86))')
    late_premium = st.text_input('Late premium (Numeric Value)')
    
    
    #converting data required to make the prediction 
    if sourcing_channel == "A":
        sourcing_channel = 0
    elif sourcing_channel == "B":
        sourcing_channel = 1
    elif sourcing_channel == "C":
        sourcing_channel = 2
    elif sourcing_channel == "D":
        sourcing_channel = 3
    elif sourcing_channel == "E":
        sourcing_channel = 4
    
    if residence_area_type == "Rural":
        residence_area_type = 0
    elif residence_area_type == "Urban":
        residence_area_type = 1
    
    
    #output prediction
    output = ''
    
    #button for prediction 
    if st.button('Predict claim'):
        
        #check if input is numeric value
        lst = [Income, application_underwriting_score, no_of_premiums_paid, sourcing_channel, residence_area_type, age, late_premium]
        numeric_value = 0
        for item in lst:
            if item.isnumeric():
                continue
            else:
                st.write("Item does not contain a numeric value.")
                break
        
        
        output = claim_prediction([Income, application_underwriting_score, no_of_premiums_paid, sourcing_channel, residence_area_type, age, late_premium])
    
    #print output
    st.success(output)


if __name__ == '__main__':
    main()
    
    

