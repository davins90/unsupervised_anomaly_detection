def main():
    import streamlit as st
    
    st.markdown("## Retraining Personas model")
    st.warning("For the customer clustering part, it was decided not to implement the retrain part of the model (recalculation of the clusters) since a more present 'human component' is essential for this phase when evaluating the result, alongside the normal evaluation metrics about the goodness of the algorithm. To be built in future development.")