import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from src.agents.medical_llm_agent import MedicalLLMAgent
from src.agents.report_generation_agent import ReportGenerationAgent
from src.database.db_manager import DatabaseManager

# Initialize agents
llm_agent = MedicalLLMAgent()
report_agent = ReportGenerationAgent()

# Initialize database
db = DatabaseManager()

# Set page config
st.set_page_config(
    page_title="Medical Report Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            height: 3em;
            font-size: 1.2em;
            border-radius: 10px;
            font-weight: bold;
        }
        .css-1v0mbdj {
            width: 100%;
        }
        .stPlotlyChart {
            margin: auto;
        }
        h1 {
            color: #2c3e50;
            font-size: 2.5em !important;
            font-weight: bold !important;
            margin-bottom: 1em !important;
        }
        h2 {
            color: #34495e;
            font-size: 2em !important;
            font-weight: bold !important;
            margin-top: 1em !important;
        }
        h3 {
            color: #2980b9;
            font-size: 1.5em !important;
        }
        .stAlert {
            font-size: 1.2em !important;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2980b9;
        }
        .metric-label {
            font-size: 1.2em;
            color: #7f8c8d;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar
with st.sidebar:
    st.title("🏥 Medical Report Analyzer")
    st.markdown("---")
    page = st.radio("Navigation", ["Upload & Analyze", "Report Generation", "Quick Summary", "Report History"])

def show_upload_and_analyze():
    st.title("Medical Data Analysis 🔬")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload Medical Data CSV", type=['csv'])
    
    if uploaded_file:
        try:
            # Save uploaded file
            save_dir = "uploads"
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.current_file = file_path
            
            # Read and display data
            df = pd.read_csv(file_path)
            st.session_state.df = df
            
            # Display data overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Records</div>
                    </div>
                """.format(len(df)), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Features</div>
                    </div>
                """.format(len(df.columns)), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Numeric Cols</div>
                    </div>
                """.format(len(df.select_dtypes(include=['int64', 'float64']).columns)), 
                unsafe_allow_html=True)
            
            with col4:
                missing = df.isnull().sum().sum()
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Missing Values</div>
                    </div>
                """.format(missing), unsafe_allow_html=True)
            
            st.markdown("### Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Analysis section
            st.markdown("## Analysis Options")
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Basic Analysis", "Detailed Analysis with Predictions"],
                help="Choose the type of analysis to perform on your data"
            )
            
            if st.button("Run Analysis", type="primary"):
                with st.spinner("Analyzing data..."):
                    # Run analysis
                    analysis_results = llm_agent.analyze(df, analysis_type)
                    st.session_state.analysis_results = analysis_results
                    
                    # Display results in tabs
                    tab1, tab2, tab3 = st.tabs(["📊 Insights", "📈 Visualizations", "🎯 Predictions"])
                    
                    with tab1:
                        if "insights" in analysis_results:
                            for insight in analysis_results["insights"]:
                                st.markdown(f"• {insight}")
                        
                        if "risk_factors" in analysis_results:
                            st.markdown("### Risk Factors")
                            for risk in analysis_results["risk_factors"]:
                                st.markdown(f"• {risk}")
                    
                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### Distribution Analysis")
                            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                            selected_col = st.selectbox("Select feature", numeric_cols)
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(data=df, x=selected_col, kde=True)
                            plt.title(f'Distribution of {selected_col}')
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            st.markdown("#### Correlation Analysis")
                            numeric_df = df.select_dtypes(include=['int64', 'float64'])
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
                            plt.title('Feature Correlations')
                            st.pyplot(fig)
                            plt.close()
                    
                    with tab3:
                        if "predictions" in analysis_results:
                            st.markdown("### Prediction Results")
                            accuracy = analysis_results.get('accuracy', 'N/A')
                            if accuracy != 'N/A':
                                st.markdown(f"Model Accuracy: {accuracy:.2%}")
                            
                            if "confusion_matrix" in analysis_results:
                                st.markdown("#### Confusion Matrix")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(analysis_results["confusion_matrix"], 
                                          annot=True, fmt='d', cmap='Blues')
                                plt.title('Confusion Matrix')
                                st.pyplot(fig)
                                plt.close()
                        else:
                            st.info("No prediction results available for this analysis.")
                    
                    st.success("Analysis complete! 🎉")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def show_report_generation():
    st.title("Report Generation 📄")
    
    if st.session_state.current_file is None:
        st.warning("Please upload a file first in the Upload & Analyze section")
        return
        
    if st.session_state.analysis_results is None:
        st.warning("Please run analysis first in the Upload & Analyze section")
        return
    
    # Show live visualizations in dashboard
    if st.session_state.current_file:
        df = pd.read_csv(st.session_state.current_file)
        
        st.markdown("## Data Insights Dashboard")
        
        # Create tabs for different visualizations
        viz_tabs = st.tabs(["📊 Distributions", "🔄 Correlations", "📈 Statistics"])
        
        with viz_tabs[0]:
            st.markdown("### Distribution Analysis")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            col1, col2 = st.columns([1, 2])
            with col1:
                selected_col = st.selectbox("Select feature", numeric_cols)
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x=selected_col, kde=True)
                plt.title(f'Distribution of {selected_col}')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with viz_tabs[1]:
            st.markdown("### Correlation Analysis")
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Feature Correlation Analysis')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with viz_tabs[2]:
            st.markdown("### Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
    
    # Report generation section
    st.markdown("## Generate Report")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        report_options = st.multiselect(
            "Select Report Sections",
            ["Executive Summary", "Data Overview", "Key Insights", 
             "Visualizations", "Statistical Analysis", "Predictions", "Recommendations"],
            default=["Executive Summary", "Data Overview", "Key Insights", "Visualizations"],
            help="Choose which sections to include in your report"
        )
    
    with col2:
        report_format = st.radio(
            "Report Format",
            ["Standard", "Detailed"],
            help="Choose the level of detail for your report"
        )
    
    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating comprehensive report..."):
            try:
                report_path = report_agent.generate_report(
                    data_file=st.session_state.current_file,
                    analysis_results=st.session_state.analysis_results,
                    report_type=report_format.lower()
                )
                
                st.success("Report generated successfully! 🎉")
                
                # Create a download button
                with open(report_path, "rb") as file:
                    btn = st.download_button(
                        label="📥 Download Report",
                        data=file,
                        file_name=os.path.basename(report_path),
                        mime="application/pdf",
                        help="Click to download the generated report"
                    )
                
                # Show a preview section
                with st.expander("Report Preview", expanded=True):
                    st.info("This is a preview of the report content. Download the PDF for the full formatted report.")
                    st.markdown(f"""
                    ### Medical Data Analysis Report
                    - Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                    - File analyzed: {os.path.basename(st.session_state.current_file)}
                    - Report type: {report_format}
                    
                    #### Included Sections:
                    {', '.join(report_options)}
                    
                    #### Key Highlights:
                    - Total records analyzed: {len(st.session_state.df):,}
                    - Features analyzed: {len(st.session_state.df.columns)}
                    - Analysis depth: {report_format}
                    
                    Download the PDF to view the complete analysis with all visualizations and detailed insights.
                    """)
                    
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                st.error("Please try again.")

def show_quick_summary():
    st.title("Quick Summary 📋")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Medical Data", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if df is not None:
            # Basic dataset statistics
            st.markdown("### Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Quick insights
            st.markdown("### Quick Insights")
            
            # Generate quick summary using LLM
            analysis_results = llm_agent.analyze(df, "basic")
            
            if "error" not in analysis_results:
                # Display insights
                if "insights" in analysis_results:
                    for insight in analysis_results["insights"]:
                        st.info(insight)
                
                # Save to database
                metadata = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "file_type": uploaded_file.type
                }
                
                report_id = db.save_report(
                    filename=uploaded_file.name,
                    report_path="",  # No report file for quick summary
                    report_type="quick_summary",
                    analysis_results=analysis_results,
                    metadata=metadata
                )
                
                # Save summary
                summary_text = "\n".join(analysis_results.get("insights", []))
                db.save_quick_summary(report_id, summary_text)
                
                st.success("Summary saved to database! 🎉")
            else:
                st.error("Failed to generate summary. Please try again.")

def show_report_history():
    st.title("Report History 📚")
    
    # Search box
    search_query = st.text_input("Search Reports", "")
    
    if search_query:
        reports = db.search_reports(search_query)
    else:
        reports = db.get_recent_reports(10)
    
    # Display reports
    for report in reports:
        with st.expander(f"{report['filename']} - {report['created_at']}"):
            st.json(report['metadata'])
            
            # Get full report details
            details = db.get_report_details(report['id'])
            if details and 'summaries' in details:
                st.markdown("### Quick Summaries")
                for summary in details['summaries']:
                    st.info(summary['summary_text'])
            
            if st.button(f"View Full Report", key=f"view_{report['id']}"):
                st.session_state.selected_report = report['id']
    
    # Display report stats
    stats = db.get_report_summary_stats()
    st.sidebar.markdown("### Report Statistics")
    st.sidebar.metric("Total Reports", stats['total_reports'])
    st.sidebar.metric("Recent Reports (24h)", stats['recent_reports'])

# Main app logic
if page == "Upload & Analyze":
    show_upload_and_analyze()
elif page == "Report Generation":
    show_report_generation()
elif page == "Quick Summary":
    show_quick_summary()
else:  # Report History
    show_report_history()
