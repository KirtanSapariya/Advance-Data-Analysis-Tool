
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import base64
from pipeline_history import PipelineHistory

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.widgetbase import Widget
    HAS_REPORTLAB = True
except ImportError:
    try:
        from fpdf import FPDF
        HAS_FPDF = True
        HAS_REPORTLAB = False
    except ImportError:
        HAS_REPORTLAB = False
        HAS_FPDF = False

class ReportGenerator:
    def __init__(self):
        self.history = PipelineHistory()

    def render_report_ui(self):
        """Render the report generation interface"""

        st.subheader("📋 Generate Comprehensive Report")

        if not HAS_REPORTLAB and not HAS_FPDF:
            st.error("📄 PDF generation libraries not available. Please install reportlab or fpdf2:")
            st.code("pip install reportlab")
            st.write("or")
            st.code("pip install fpdf2")
            return

        # Report configuration
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Report Sections")

            # Section selection
            sections = {
                'data_summary': st.checkbox("📈 Data Summary", value=True),
                'cleaning_steps': st.checkbox("🧹 Cleaning Steps", value=True),
                'validation': st.checkbox("✅ Validation Results", value=True),
                'scaling_summary': st.checkbox("⚖️ Scaling Summary", value=True),
                'model_training': st.checkbox("🤖 Model Training & Metrics", value=True),
                'visualizations': st.checkbox("📊 Visualizations", value=True),
                'pipeline_history': st.checkbox("📚 Pipeline History", value=True)
            }

        with col2:
            st.markdown("### ⚙️ Report Settings")

            # Report settings
            report_title = st.text_input("Report Title", value="Data Analysis & Pipeline Report")
            dataset_name = st.selectbox("Primary Dataset", list(st.session_state.datasets.keys()) if st.session_state.datasets else ["No datasets"])
            author_name = st.text_input("Author Name", value="Data Science Team")

            # Chart selection
            if st.session_state.get('charts', []):
                selected_charts = st.multiselect(
                    "Select Charts to Include",
                    [f"Chart {i+1}: {chart.get('chart_type', 'Unknown')}" for i, chart in enumerate(st.session_state.charts)],
                    default=[f"Chart {i+1}: {chart.get('chart_type', 'Unknown')}" for i, chart in enumerate(st.session_state.charts[:3])]
                )
            else:
                selected_charts = []
                st.info("No charts available. Create some visualizations first.")

        # Preview and generation
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👀 Preview Report Content", type="secondary"):
                self._render_report_preview(sections, dataset_name, selected_charts)

        with col2:
            if st.button("📄 Generate PDF Report", type="primary"):
                if dataset_name != "No datasets":
                    self._generate_pdf_report(
                        sections, report_title, dataset_name,
                        author_name, selected_charts
                    )
                else:
                    st.error("Please select a valid dataset for the report.")

    def _render_report_preview(self, sections, dataset_name, selected_charts):
        """Render a preview of the report content"""

        st.markdown("## 📋 Report Content Preview")

        if dataset_name == "No datasets":
            st.warning("No dataset selected for preview.")
            return

        df = st.session_state.datasets[dataset_name]

        # Data Summary
        if sections['data_summary']:
            st.markdown("### 📊 Data Summary")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Duplicates", df.duplicated().sum())

            st.dataframe(df.head(), use_container_width=True)

        # Cleaning Steps
        if sections['cleaning_steps']:
            st.markdown("### 🧹 Data Cleaning Steps")

            cleaning_steps = [step for step in st.session_state.pipeline_history
                            if step['operation'] in ['Automatic Cleaning', 'Missing Value Treatment', 'Duplicate Removal', 'Outlier Handling']]

            if cleaning_steps:
                for step in cleaning_steps[-5:]:  # Show last 5 cleaning steps
                    st.write(f"✅ **{step['operation']}**: {step['description']}")
            else:
                st.info("No cleaning steps recorded in pipeline history.")

        # Validation Results
        if sections['validation']:
            st.markdown("### ✅ Validation Results")

            # Calculate basic validation metrics
            completeness = ((df.count().sum()) / (len(df) * len(df.columns))) * 100
            st.write(f"**Data Completeness**: {completeness:.1f}%")

            uniqueness = df.nunique().sum() / len(df.columns)
            st.write(f"**Average Uniqueness**: {uniqueness:.1f} unique values per column")

        # Model Training Results
        if sections['model_training'] and 'training_results' in st.session_state:
            st.markdown("### 🤖 Model Training Results")

            results = st.session_state.training_results
            if results:
                best_model = max(results, key=lambda x: x['metrics'].get('accuracy', x['metrics'].get('r2', 0)))
                st.write(f"**Best Model**: {best_model['model_name']}")

                if best_model['task_type'] == 'Classification':
                    st.write(f"**Accuracy**: {best_model['metrics'].get('accuracy', 0):.4f}")
                elif best_model['task_type'] == 'Regression':
                    st.write(f"**R² Score**: {best_model['metrics'].get('r2', 0):.4f}")

        # Charts
        if sections['visualizations'] and selected_charts:
            st.markdown("### 📊 Visualizations")
            st.write(f"Selected {len(selected_charts)} charts for inclusion in report.")

        # Pipeline History
        if sections['pipeline_history']:
            st.markdown("### 📚 Pipeline History")

            history_summary = self.history.get_history_summary()
            if history_summary:
                st.write(f"**Total Steps**: {history_summary['total_steps']}")
                st.write(f"**Successful Steps**: {history_summary['successful_steps']}")
                st.write(f"**Failed Steps**: {history_summary['failed_steps']}")

    def _generate_pdf_report(self, sections, title, dataset_name, author, selected_charts):
        """Generate comprehensive PDF report"""

        try:
            with st.spinner("Generating PDF report..."):

                if HAS_REPORTLAB:
                    pdf_buffer = self._generate_reportlab_pdf(
                        sections, title, dataset_name, author, selected_charts
                    )
                    library_used = "ReportLab"
                elif HAS_FPDF:
                    pdf_buffer = self._generate_fpdf_pdf(
                        sections, title, dataset_name, author, selected_charts
                    )
                    library_used = "FPDF"
                else:
                    st.error("No PDF generation library available.")
                    return

                if pdf_buffer:
                    # Create download button
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"data_analysis_report_{timestamp}.pdf"

                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=filename,
                        mime="application/pdf"
                    )

                    st.success(f"✅ PDF report generated successfully using {library_used}!")

                    # Log to history
                    self.history.log_step(
                        "Report Generation",
                        f"Generated PDF report: {title}",
                        {
                            "sections": list(sections.keys()),
                            "dataset": dataset_name,
                            "author": author,
                            "charts_included": len(selected_charts)
                        },
                        "success"
                    )

        except Exception as e:
            st.error(f"Error generating PDF report: {str(e)}")
            self.history.log_step(
                "Report Generation",
                "Failed to generate PDF report",
                {"error": str(e)},
                "error"
            )

    def _generate_reportlab_pdf(self, sections, title, dataset_name, author, selected_charts):
        """Generate PDF using ReportLab"""

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2E86AB')
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#A23B72')
        )

        # Title page
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Dataset: {dataset_name}", styles['Normal']))
        story.append(Paragraph(f"Author: {author}", styles['Normal']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 30))

        # Get dataset
        df = st.session_state.datasets[dataset_name]

        # Table of Contents
        story.append(Paragraph("Table of Contents", heading_style))
        toc_data = []

        section_names = {
            'data_summary': 'Data Summary',
            'cleaning_steps': 'Data Cleaning Steps',
            'validation': 'Validation Results',
            'scaling_summary': 'Scaling & Encoding Summary',
            'model_training': 'Model Training Results',
            'visualizations': 'Visualizations',
            'pipeline_history': 'Pipeline History'
        }

        for key, value in sections.items():
            if value:
                toc_data.append([section_names[key], ""])

        if toc_data:
            toc_table = Table(toc_data)
            toc_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            story.append(toc_table)

        story.append(PageBreak())

        # Data Summary Section
        if sections['data_summary']:
            story.append(Paragraph("Data Summary", heading_style))

            # Basic statistics
            summary_data = [
                ['Metric', 'Value'],
                ['Total Rows', str(len(df))],
                ['Total Columns', str(len(df.columns))],
                ['Missing Values', str(df.isnull().sum().sum())],
                ['Duplicate Rows', str(df.duplicated().sum())],
                ['Memory Usage (MB)', f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f}"]
            ]

            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 12))

            # Column information
            story.append(Paragraph("Column Information", styles['Heading3']))

            col_data = [['Column Name', 'Data Type', 'Non-Null Count', 'Unique Values']]

            for col in df.columns:
                col_data.append([
                    col,
                    str(df[col].dtype),
                    str(df[col].count()),
                    str(df[col].nunique())
                ])

            # Limit to first 20 columns for space
            if len(col_data) > 21:  # 20 data rows + header
                col_data = col_data[:21]
                col_data.append(['...', '...', '...', '...'])

            col_table = Table(col_data)
            col_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(col_table)
            story.append(PageBreak())

        # Data Cleaning Steps
        if sections['cleaning_steps']:
            story.append(Paragraph("Data Cleaning Steps", heading_style))

            cleaning_steps = [step for step in st.session_state.pipeline_history
                            if step['operation'] in ['Automatic Cleaning', 'Missing Value Treatment',
                                                   'Duplicate Removal', 'Outlier Handling', 'Custom Transformation']]

            if cleaning_steps:
                cleaning_data = [['Step', 'Operation', 'Description', 'Status']]

                for i, step in enumerate(cleaning_steps, 1):
                    cleaning_data.append([
                        str(i),
                        step['operation'],
                        step['description'][:100] + '...' if len(step['description']) > 100 else step['description'],
                        step['status'].title()
                    ])

                cleaning_table = Table(cleaning_data, colWidths=[0.5*inch, 1.5*inch, 3*inch, 0.8*inch])
                cleaning_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                story.append(cleaning_table)
            else:
                story.append(Paragraph("No cleaning steps recorded in pipeline history.", styles['Normal']))

            story.append(PageBreak())

        # Validation Results
        if sections['validation']:
            story.append(Paragraph("Validation Results", heading_style))

            # Calculate validation metrics
            total_cells = df.size
            null_cells = df.isnull().sum().sum()
            completeness = ((total_cells - null_cells) / total_cells) * 100 if total_cells > 0 else 0

            # Data quality metrics
            quality_data = [
                ['Quality Metric', 'Score', 'Status'],
                ['Data Completeness', f"{completeness:.1f}%", 'Good' if completeness > 90 else 'Needs Attention'],
                ['Duplicate Rate', f"{(df.duplicated().sum()/len(df)*100):.1f}%", 'Good' if df.duplicated().sum()/len(df) < 0.05 else 'Review'],
                ['Column Consistency', 'Analyzed', 'See Details'],
            ]

            quality_table = Table(quality_data)
            quality_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(quality_table)
            story.append(PageBreak())

        # Model Training Results
        if sections['model_training'] and 'training_results' in st.session_state:
            story.append(Paragraph("Model Training Results", heading_style))

            results = st.session_state.training_results
            if results:
                # Model comparison table
                model_data = [['Model', 'Task Type', 'Primary Metric', 'Score', 'Training Time (s)']]

                for result in results:
                    metrics = result['metrics']

                    if result['task_type'] == 'Classification':
                        primary_metric = 'Accuracy'
                        score = f"{metrics.get('accuracy', 0):.4f}"
                    elif result['task_type'] == 'Regression':
                        primary_metric = 'R² Score'
                        score = f"{metrics.get('r2', 0):.4f}"
                    else:
                        primary_metric = 'N/A'
                        score = 'N/A'

                    model_data.append([
                        result['model_name'],
                        result['task_type'],
                        primary_metric,
                        score,
                        f"{result['training_time']:.2f}"
                    ])

                model_table = Table(model_data)
                model_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(model_table)

                # Best model details
                if results:
                    best_model = max(results, key=lambda x: x['metrics'].get('accuracy', x['metrics'].get('r2', 0)))
                    story.append(Spacer(1, 12))
                    story.append(Paragraph("Best Performing Model", styles['Heading3']))
                    story.append(Paragraph(f"Model: {best_model['model_name']}", styles['Normal']))
                    story.append(Paragraph(f"Task: {best_model['task_type']}", styles['Normal']))

                    if best_model['task_type'] == 'Classification':
                        story.append(Paragraph(f"Accuracy: {best_model['metrics'].get('accuracy', 0):.4f}", styles['Normal']))
                        story.append(Paragraph(f"Precision: {best_model['metrics'].get('precision', 0):.4f}", styles['Normal']))
                        story.append(Paragraph(f"Recall: {best_model['metrics'].get('recall', 0):.4f}", styles['Normal']))
                        story.append(Paragraph(f"F1-Score: {best_model['metrics'].get('f1', 0):.4f}", styles['Normal']))
                    elif best_model['task_type'] == 'Regression':
                        story.append(Paragraph(f"R² Score: {best_model['metrics'].get('r2', 0):.4f}", styles['Normal']))
                        story.append(Paragraph(f"RMSE: {best_model['metrics'].get('rmse', 0):.4f}", styles['Normal']))
                        story.append(Paragraph(f"MAE: {best_model['metrics'].get('mae', 0):.4f}", styles['Normal']))
            else:
                story.append(Paragraph("No model training results available.", styles['Normal']))

            story.append(PageBreak())

        # Pipeline History
        if sections['pipeline_history']:
            story.append(Paragraph("Pipeline History", heading_style))

            history_summary = self.history.get_history_summary()
            if history_summary:
                # Summary statistics
                history_stats = [
                    ['Pipeline Summary', 'Value'],
                    ['Total Steps', str(history_summary['total_steps'])],
                    ['Successful Steps', str(history_summary['successful_steps'])],
                    ['Failed Steps', str(history_summary['failed_steps'])],
                    ['Start Time', history_summary['start_time'][:19] if 'start_time' in history_summary else 'N/A'],
                    ['End Time', history_summary['end_time'][:19] if 'end_time' in history_summary else 'N/A']
                ]

                history_table = Table(history_stats)
                history_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(history_table)

                # Recent steps
                story.append(Spacer(1, 12))
                story.append(Paragraph("Recent Pipeline Steps", styles['Heading3']))

                recent_steps = history_summary.get('steps', [])[-10:]  # Last 10 steps
                if recent_steps:
                    steps_data = [['Step', 'Operation', 'Description', 'Status']]

                    for i, step in enumerate(recent_steps, 1):
                        steps_data.append([
                            str(i),
                            step['operation'],
                            step['description'][:80] + '...' if len(step['description']) > 80 else step['description'],
                            step['status'].title()
                        ])

                    steps_table = Table(steps_data, colWidths=[0.5*inch, 1.2*inch, 3.5*inch, 0.8*inch])
                    steps_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ]))
                    story.append(steps_table)
            else:
                story.append(Paragraph("No pipeline history available.", styles['Normal']))

        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph("--- End of Report ---", styles['Normal']))
        story.append(Paragraph(f"Generated by Data Analysis & Pipeline Tool on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

    def _generate_fpdf_pdf(self, sections, title, dataset_name, author, selected_charts):
        """Generate PDF using FPDF (fallback option)"""

        buffer = io.BytesIO()

        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 15)
                self.cell(0, 10, title, 0, 1, 'C')
                self.ln(10)

            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        pdf = PDF()
        pdf.add_page()
        pdf.set_font('Arial', size=12)

        # Title page
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, title, 0, 1, 'C')
        pdf.ln(5)

        pdf.set_font('Arial', size=12)
        pdf.cell(0, 10, f'Dataset: {dataset_name}', 0, 1)
        pdf.cell(0, 10, f'Author: {author}', 0, 1)
        pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
        pdf.ln(10)

        # Get dataset
        df = st.session_state.datasets[dataset_name]

        # Data Summary Section
        if sections['data_summary']:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Data Summary', 0, 1)
            pdf.ln(5)

            pdf.set_font('Arial', size=10)
            pdf.cell(0, 5, f'Total Rows: {len(df)}', 0, 1)
            pdf.cell(0, 5, f'Total Columns: {len(df.columns)}', 0, 1)
            pdf.cell(0, 5, f'Missing Values: {df.isnull().sum().sum()}', 0, 1)
            pdf.cell(0, 5, f'Duplicate Rows: {df.duplicated().sum()}', 0, 1)
            pdf.ln(10)

        # Data Cleaning Steps
        if sections['cleaning_steps']:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Data Cleaning Steps', 0, 1)
            pdf.ln(5)

            cleaning_steps = [step for step in st.session_state.pipeline_history
                            if step['operation'] in ['Automatic Cleaning', 'Missing Value Treatment',
                                                   'Duplicate Removal', 'Outlier Handling']]

            pdf.set_font('Arial', size=10)
            if cleaning_steps:
                for i, step in enumerate(cleaning_steps[-5:], 1):  # Last 5 steps
                    pdf.cell(0, 5, f'{i}. {step["operation"]}: {step["description"][:80]}...', 0, 1)
            else:
                pdf.cell(0, 5, 'No cleaning steps recorded.', 0, 1)
            pdf.ln(10)

        # Model Training Results
        if sections['model_training'] and 'training_results' in st.session_state:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Model Training Results', 0, 1)
            pdf.ln(5)

            results = st.session_state.training_results
            if results:
                pdf.set_font('Arial', size=10)
                for result in results:
                    metrics = result['metrics']

                    pdf.cell(0, 5, f'Model: {result["model_name"]}', 0, 1)
                    pdf.cell(0, 5, f'Task: {result["task_type"]}', 0, 1)

                    if result['task_type'] == 'Classification':
                        pdf.cell(0, 5, f'Accuracy: {metrics.get("accuracy", 0):.4f}', 0, 1)
                    elif result['task_type'] == 'Regression':
                        pdf.cell(0, 5, f'R² Score: {metrics.get("r2", 0):.4f}', 0, 1)

                    pdf.cell(0, 5, f'Training Time: {result["training_time"]:.2f}s', 0, 1)
                    pdf.ln(5)

        # Pipeline History Summary
        if sections['pipeline_history']:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Pipeline History Summary', 0, 1)
            pdf.ln(5)

            history_summary = self.history.get_history_summary()
            if history_summary:
                pdf.set_font('Arial', size=10)
                pdf.cell(0, 5, f'Total Steps: {history_summary["total_steps"]}', 0, 1)
                pdf.cell(0, 5, f'Successful Steps: {history_summary["successful_steps"]}', 0, 1)
                pdf.cell(0, 5, f'Failed Steps: {history_summary["failed_steps"]}', 0, 1)

        # Save to buffer
        pdf_output = pdf.output(dest='S').encode('latin-1')
        buffer.write(pdf_output)
        buffer.seek(0)
        return buffer

    def _save_chart_as_image(self, chart_fig, width=800, height=600):
        """Convert plotly figure to image bytes for PDF inclusion"""
        try:
            img_bytes = chart_fig.to_image(format="png", width=width, height=height)
            return img_bytes
        except Exception as e:
            st.warning(f"Could not convert chart to image: {str(e)}")
            return None

    def export_pipeline_data(self):
        """Export all pipeline data for backup/sharing"""

        st.markdown("### 📦 Export Pipeline Data")

        export_data = {
            'datasets': {},  # Can't serialize pandas DataFrames directly
            'pipeline_history': st.session_state.get('pipeline_history', []),
            'training_results': [],  # Can't serialize sklearn models directly
            'charts_metadata': []
        }

        # Export dataset summaries (not full data due to size)
        for name, df in st.session_state.get('datasets', {}).items():
            export_data['datasets'][name] = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'missing_counts': df.isnull().sum().to_dict(),
                'summary_stats': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
            }

        # Export model training summaries (not full models)
        for result in st.session_state.get('training_results', []):
            training_summary = {
                'model_name': result['model_name'],
                'task_type': result['task_type'],
                'training_time': result['training_time'],
                'feature_columns': result['feature_columns'],
                'target_column': result.get('target_column'),
                'timestamp': result['timestamp'],
                'metrics_summary': {}
            }

            # Include key metrics only
            metrics = result['metrics']
            if result['task_type'] == 'Classification':
                training_summary['metrics_summary'] = {
                    'accuracy': metrics.get('accuracy'),
                    'precision': metrics.get('precision'),
                    'recall': metrics.get('recall'),
                    'f1': metrics.get('f1')
                }
            elif result['task_type'] == 'Regression':
                training_summary['metrics_summary'] = {
                    'r2': metrics.get('r2'),
                    'rmse': metrics.get('rmse'),
                    'mae': metrics.get('mae'),
                    'mse': metrics.get('mse')
                }

            export_data['training_results'].append(training_summary)

        # Export chart metadata
        for i, chart in enumerate(st.session_state.get('charts', [])):
            chart_metadata = {
                'chart_id': chart.get('chart_id', f'chart_{i}'),
                'dataset': chart.get('dataset'),
                'chart_type': chart.get('chart_type'),
                'x_column': chart.get('x_column'),
                'y_column': chart.get('y_column'),
                'config': chart.get('config', {})
            }
            export_data['charts_metadata'].append(chart_metadata)

        # Generate JSON
        import json
        export_json = json.dumps(export_data, indent=2, default=str)

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📥 Download Pipeline Summary (JSON)",
                data=export_json,
                file_name=f"pipeline_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        with col2:
            # Also create a CSV summary
            pipeline_summary = []

            for step in st.session_state.get('pipeline_history', []):
                pipeline_summary.append({
                    'timestamp': step['timestamp'],
                    'operation': step['operation'],
                    'description': step['description'],
                    'status': step['status']
                })

            if pipeline_summary:
                summary_df = pd.DataFrame(pipeline_summary)
                summary_csv = summary_df.to_csv(index=False)

                st.download_button(
                    label="📊 Download Pipeline History (CSV)",
                    data=summary_csv,
                    file_name=f"pipeline_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        # Statistics
        st.markdown("**Export Summary:**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Datasets", len(export_data['datasets']))

        with col2:
            st.metric("Pipeline Steps", len(export_data['pipeline_history']))

        with col3:
            st.metric("Models Trained", len(export_data['training_results']))

        with col4:
            st.metric("Charts Created", len(export_data['charts_metadata']))
