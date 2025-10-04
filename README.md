# 📊 Advanced Data Processing & Visualization Tool

An interactive data processing and visualization tool built with Streamlit. Effortlessly clean, explore, validate, and visualize datasets using powerful charts, advanced analytics, and automated data auditing—no coding required. Ideal for rapid EDA and reporting.

## 🚀 Features

### 📈 **Data Visualization**
- **16+ Chart Types**: Bar, Line, Scatter, 3D Scatter, Heatmap, Treemap, Sankey, Radar, etc.
- **Interactive Charts**: Built with Plotly for dynamic exploration
- **Advanced Analytics**: Correlation analysis, distribution matrices, parallel coordinates
- **Automated Insights**: AI-powered data audit and recommendations

### 🧹 **Data Preprocessing** 
- **Automatic Cleaning**: Smart duplicate removal, missing value imputation, data type inference
- **Manual Cleaning**: Step-by-step control over missing values, outliers, and transformations
- **Feature Engineering**: Scaling, encoding, custom transformations
- **Safe Processing**: Handles complex data types (lists, dicts) without errors

### 🔍 **Data Validation**
- **Quality Metrics**: Completeness, accuracy, consistency analysis  
- **Schema Validation**: Data type and format checking
- **Business Rules**: Custom validation logic
- **Automated Reports**: Comprehensive data quality assessments

### 🔌 **Data Connection**
- **Multiple Sources**: CSV, Excel, JSON, databases
- **Cloud Storage**: AWS S3, Google Drive integration
- **Real-time Loading**: Progress tracking with detailed feedback
- **Format Detection**: Automatic parsing and encoding handling

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/advanced-data-tool.git
cd advanced-data-tool

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📋 Dependencies

Core libraries used:
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations  
- **Scikit-learn**: Machine learning preprocessing
- **NumPy**: Numerical computations
- **Scipy**: Statistical functions

See `requirements.txt` for complete dependency list.

## 🎯 Usage

### Getting Started
1. **Launch the app**: Run `streamlit run app.py`
2. **Load your data**: Use the "Data Connection" tab to upload files
3. **Explore visually**: Switch to "Raw Data Visualization" for charts
4. **Clean your data**: Use "Data Analysis & Preprocessing" for cleaning
5. **Validate quality**: Check "Data Validation" for quality metrics
6. **View history**: Track all operations in "Pipeline History"

### Sample Workflows

#### **Quick Data Exploration**
```
1. Upload CSV file
2. Go to Data Overview → see column info and missing values
3. Create charts → select chart type and columns  
4. Generate insights → use Auto Audit feature
```

#### **Data Cleaning Pipeline**
```
1. Load raw dataset
2. Run Automatic Clean → review suggested operations
3. Apply transformations → create cleaned dataset
4. Validate results → check data quality metrics
5. Export cleaned data → download processed dataset
```

#### **Advanced Analytics**
```
1. Load preprocessed data
2. Create correlation heatmap → identify relationships
3. Build 3D scatter plots → explore multidimensional patterns
4. Generate parallel coordinates → analyze feature interactions
```

## 📁 Project Structure

```
advanced-data-tool/
├── app.py                      # Main Streamlit application
├── data_connection.py          # Data loading and connection utilities
├── visualization_safe.py       # Chart generation and plotting (safe version)
├── data_preprocessing_safe.py  # Data cleaning and preprocessing (safe version)  
├── validation.py              # Data validation and quality checks
├── pipeline_history.py        # Operation history tracking
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                # Git ignore rules
├── LICENSE                   # Project license
└── assets/                   # Static assets and examples
    └── sample_data.csv       # Sample dataset for testing
```

## 🔧 Configuration

The application uses session state for data persistence and supports:

- **Memory Management**: Efficient handling of large datasets
- **Error Recovery**: Robust error handling for edge cases
- **Progress Tracking**: Real-time feedback for long operations
- **History Logging**: Complete audit trail of all operations

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/advanced-data-tool.git

# Install development dependencies
pip install -r requirements.txt

# Run tests (when available)
python -m pytest tests/

# Start development server
streamlit run app.py --server.runOnSave true
```

## 📊 Chart Types Supported

### Categorical Data
- Bar Chart, Column Chart
- Pie Chart, Donut Chart  
- Stacked Bar, Grouped Bar
- Lollipop Chart
- Treemap, Sunburst
- Sankey Diagram

### Numerical Data
- Histogram, Box Plot, Violin Plot
- Line Chart, Area Chart
- Scatter Plot, Bubble Chart
- 3D Scatter Plot
- Contour Plot, Heatmap
- Parallel Coordinates, Radar Chart

## 🧪 Advanced Features

### **Automated Data Audit**
- Missing value analysis
- Duplicate detection  
- Outlier identification
- Data type recommendations
- Quality scoring

### **Smart Preprocessing**
- Automatic scaling (StandardScaler, RobustScaler, etc.)
- Intelligent encoding (OneHot, Ordinal, Label)
- Missing value strategies (mean, median, KNN, iterative)
- Outlier handling (IQR, Z-score, Isolation Forest)

### **Export Capabilities**
- Processed datasets (CSV, Excel)
- Transformation pipelines (JSON)
- Visualization exports (PNG, HTML)
- Quality reports (PDF)

## 🐛 Troubleshooting

### Common Issues

**Issue**: "ModuleNotFoundError" when running the app
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Charts not displaying properly  
**Solution**: Update your browser and ensure JavaScript is enabled

**Issue**: Large files causing memory errors
**Solution**: Process files in chunks or use a machine with more RAM

**Issue**: "unhashable type: 'list'" errors
**Solution**: The app automatically converts complex data types to strings

### Performance Tips
- Use sampling for large datasets (>100MB)
- Close unused tabs to free memory
- Process files locally for faster performance
- Use CSV format for optimal loading speed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) framework
- Visualizations powered by [Plotly](https://plotly.com/)
- Data processing with [Pandas](https://pandas.pydata.org/) and [Scikit-learn](https://scikit-learn.org/)
- Inspired by the need for no-code data analysis tools

## 🔮 Roadmap

Future enhancements planned:
- [ ] Machine learning model integration
- [ ] Real-time data streaming support
- [ ] Advanced statistical tests
- [ ] Collaborative features
- [ ] API endpoints for programmatic access
- [ ] Docker containerization
- [ ] Cloud deployment templates

---

⭐ **Star this repository** if you find it useful!
