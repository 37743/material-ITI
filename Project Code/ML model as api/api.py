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
    income = st.selectbox('1\. Yearly Income - (Numeric Value: 0:(0-23K) , 1:(23-109k), 2:(109-194k), 3:(194-279k), 4:(279-364k), 5:(364-450k), 6:(>450k))',\
                           ("--","0", "1", "2", "3", "4", "5", "6"), key="0")
    application_underwriting_score = st.text_input('2\. Application Underwriting Score - (Numeric Value: [0-100]%)',\
                                                    max_chars=3, key="1")
    no_of_premiums_paid = st.text_input('3\. Number of Premiums Paid - (Numeric Value)', key="2")
    sourcing_channel = st.selectbox('4\. Sourcing Channel', ("--", "A", "B", "C", "D", "E"), key="3")
    residence_area_type = st.selectbox('5\. Residence Area Type', ("--", "Rural", "Urban"), key="4")
    age = st.selectbox('6\. User Age in Years - (Numeric Value: 0:(0-37), 1:(38-53), 2:(54-70), 3:(71-86), 4:(>86))',\
                        ("--","0", "1", "2", "3", "4"), key="5")
    late_premium = st.text_input('7\. Late Premium - (Numeric Value)', key="6")

    #converting data required to make the prediction
    match sourcing_channel:
        case "A":
            sourcing_channel = 0
        case "B":
            sourcing_channel = 1
        case "C":
            sourcing_channel = 2
        case "D":
            sourcing_channel = 3
        case "E":
            sourcing_channel = 4
    
    if income == "--":
        income = 0
    if residence_area_type == "Rural":
        residence_area_type = 0
    elif residence_area_type == "Urban":
        residence_area_type = 1
    
    
    #output prediction
    output = ''
    output_check = True
    def clear_text():
        for i in ["1","2","6"]: #not select boxes
            if not st.session_state[str(i)].isnumeric():
                    st.session_state[str(i)] = ""
        if not (st.session_state["1"] == ""):
            if int(st.session_state["1"]) > 100: # Percentages should NOT go over 100 
                st.session_state["1"] = ""

    #button for prediction 
    if st.button('Predict Claim', on_click=clear_text):
        output_check = True
        lst = [income, application_underwriting_score, no_of_premiums_paid, sourcing_channel, residence_area_type, age, late_premium]
        for index, item in enumerate(lst):
            if ((item == "") or (item == "--")):
                st.write("Kindly re-enter valid values at text box: {i:.0f}.".format(i=index+1))
                output_check = False
        if output_check:
            output = claim_prediction(lst)
    
    #print output
    st.success(output)
    st.image("bar.png")
    st.image("pie.png")


if __name__ == '__main__':
    main()
    
    

