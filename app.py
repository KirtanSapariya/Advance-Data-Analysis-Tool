
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import zipfile
import tempfile
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_connections import DataConnections
from data_preprocessing import DataPreprocessing
from visualization import Visualization
from modeling import ModelTraining
from validation import DataValidation
from report_generation import ReportGenerator
from pipeline_history import PipelineHistory
from sample_data import SampleDatasets

# Page configuration
st.set_page_config(
    page_title="Data Analysis & Pipeline Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}
    if 'transformers' not in st.session_state:
        st.session_state.transformers = {}
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'pipeline_history' not in st.session_state:
        st.session_state.pipeline_history = []
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    if 'charts' not in st.session_state:
        st.session_state.charts = []

def main():
    initialize_session_state()

    # Initialize helper classes
    data_conn = DataConnections()
    data_prep = DataPreprocessing()
    viz = Visualization()
    model_trainer = ModelTraining()
    validator = DataValidation()
    reporter = ReportGenerator()
    history = PipelineHistory()
    sample_data = SampleDatasets()

    # Title and sidebar
    st.title("🔬 Data Analysis & Pipeline Tool")
    st.markdown("*Production-ready data science pipeline with comprehensive analysis capabilities*")

    # Sidebar for session management
    with st.sidebar:
        st.header("📁 Session Management")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Session", help="Save current session to file"):
                session_data = {
                    'datasets': st.session_state.datasets,
                    'pipeline_history': st.session_state.pipeline_history,
                    'charts': st.session_state.charts
                }

                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(session_data, f, default=str, indent=2)
                    session_file = f.name

                with open(session_file, 'rb') as f:
                    st.download_button(
                        "⬇️ Download Session",
                        f.read(),
                        file_name=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

        with col2:
            uploaded_session = st.file_uploader("📂 Load Session", type=['json'])
            if uploaded_session:
                try:
                    session_data = json.load(uploaded_session)
                    st.session_state.datasets = session_data.get('datasets', {})
                    st.session_state.pipeline_history = session_data.get('pipeline_history', [])
                    st.session_state.charts = session_data.get('charts', [])
                    st.success("Session loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading session: {str(e)}")

        # Sample datasets
        st.header("🎯 Quick Start")
        sample_dataset = st.selectbox(
            "Load Sample Dataset",
            ["None", "Iris", "Titanic", "Housing", "Wine Quality"],
            help="Load a sample dataset to test functionality"
        )

        if sample_dataset != "None" and st.button("Load Sample"):
            df = sample_data.load_sample(sample_dataset.lower())
            if df is not None:
                dataset_name = f"sample_{sample_dataset.lower()}"
                st.session_state.datasets[dataset_name] = df
                st.session_state.current_dataset = dataset_name
                history.log_step("Data Loading", f"Loaded sample dataset: {sample_dataset}", {"rows": len(df), "columns": len(df.columns)}, "success")
                st.success(f"Loaded {sample_dataset} dataset!")
                st.rerun()

        # Current datasets
        if st.session_state.datasets:
            st.header("📊 Current Datasets")
            for name, df in st.session_state.datasets.items():
                st.write(f"**{name}**: {len(df)} rows × {len(df.columns)} columns")

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "1. Data Connection",
        "2. Raw Data Visualization",
        "3. Analysis & Preprocessing",
        "4. Validation",
        "5. Scaling & Encoding",
        "6. Modeling",
        "7. Report Generation"
    ])

    # Tab 1: Data Connection
    with tab1:
        st.header("🔌 Data Connection")
        data_conn.render_data_connection_ui()

    # Tab 2: Raw Data Visualization
    with tab2:
        st.header("📊 Raw Data Visualization")
        if st.session_state.datasets:
            viz.render_visualization_ui()
        else:
            st.info("Please load a dataset first in the Data Connection tab.")

    # Tab 3: Analysis & Preprocessing
    with tab3:
        st.header("🧹 Data Analysis & Preprocessing")
        if st.session_state.datasets:
            data_prep.render_preprocessing_ui()
        else:
            st.info("Please load a dataset first in the Data Connection tab.")

    # Tab 4: Validation
    with tab4:
        st.header("✅ Dataset Validation")
        if st.session_state.datasets:
            validator.render_validation_ui()
        else:
            st.info("Please load a dataset first in the Data Connection tab.")

    # Tab 5: Scaling & Encoding
    with tab5:
        st.header("⚖️ Scaling & Encoding")
        if st.session_state.datasets:
            data_prep.render_scaling_ui()
        else:
            st.info("Please load a dataset first in the Data Connection tab.")

    # Tab 6: Modeling
    with tab6:
        st.header("🤖 Feature Selection, Model Training & Validation")
        if st.session_state.datasets:
            model_trainer.render_modeling_ui()
        else:
            st.info("Please load a dataset first in the Data Connection tab.")

    # Tab 7: Report Generation
    with tab7:
        st.header("📋 Report Generation")
        if st.session_state.pipeline_history:
            reporter.render_report_ui()
        else:
            st.info("No pipeline history available. Please perform some operations first.")

    # Pipeline History in sidebar
    with st.sidebar:
        if st.session_state.pipeline_history:
            st.header("📚 Pipeline History")
            with st.expander("View History", expanded=False):
                history.display_history()

            if st.button("🧹 Clear History"):
                st.session_state.pipeline_history = []
                st.rerun()

if __name__ == "__main__":
    main()
