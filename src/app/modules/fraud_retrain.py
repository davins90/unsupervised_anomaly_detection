def main():
    import streamlit as st
    from ploomber import DAG
    from ploomber.tasks import NotebookRunner
    
    st.title("fraud retrain")
    
    def run_pipeline():
        dag.up()
        # Print log information
        st.write(dag.render_status())
        
        
    dag = DAG()
    
    task1 = NotebookRunner('notebooks/prod_version/2_data_preparation/2.2_data_preparation_v2.ipynb', dag = dag, product='data_lake/output_prod/train.pkl', name='notebook1')
    task2 = NotebookRunner('notebooks/prod_version/3_features_engineering/3.0_features_eng_v27_12.ipynb', dag, product='',name='notebook2')
    
    task2.inherit(task1)
    
    st.title("Notebook Pipeline")

    if st.button("Run Pipeline"):
        run_pipeline()
        st.success("Pipeline finished running")
    else:
        st.info("Click the button to run the pipeline")

    st.graphviz_chart(dag.render_graph())
