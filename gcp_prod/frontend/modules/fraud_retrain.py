def main():
    import streamlit as st
    import pickle
    import pandas as pd
    import matplotlib.pyplot as plt

    from ploomber_engine import execute_notebook

    st.markdown("## Retraining Fraud Detection model")
    st.write("In this section, the training of the classification model (which generates the fraud probability score) can be carried out again. The [Ploomber](https://ploomber.io/) library, [Ploomber Engine](https://engine.ploomber.io/en/latest/quick-start.html), was used to build this section, which is useful for recalculating the notebooks used to prepare the data and do the model training, in a pipeline similar to Apache Airflow. Intermediate outputs will be displayed on the page.")
    
    st.markdown("### Input form")
    form = st.form(key='my_form')
    train_dim = form.number_input("Select dimension of the training set (%)",step=0.5,value=0.7)
    max_depth_model = form.number_input("Max depth of the main algo: ",step=1,value=3)
    max_number_estimator = form.number_input("Max number of estimators for the main algo: ",step=50,value=200)
    val_test_dim = 0.15
    submit_button = form.form_submit_button(label='Submit')
    
    if submit_button is True:
    
        st.markdown("#### Start Data Preparation")

        out = execute_notebook("notebooks/input/2.0_data_preparation_input.ipynb",None,log_output=True,verbose=True,
                               parameters={"train_dim":train_dim,
                                           "val_test_dim":val_test_dim})

        st.markdown("#### End Data Preparation")

        st.markdown("#### Start Features Engineering")

        out2 = execute_notebook("notebooks/input/3.0_features_eng_input.ipynb",None,log_output=True,verbose=True)

        st.markdown("#### End Features Enginnering")


        st.markdown("#### Start Traininig model")

        out3 = execute_notebook("notebooks/input/4.0_training_evaluation_input.ipynb",None,log_output=True,verbose=True,
                               parameters={"max_depth_model":max_depth_model,
                                           "max_number_estimator":max_number_estimator})

        st.markdown("#### End Training model")
    

