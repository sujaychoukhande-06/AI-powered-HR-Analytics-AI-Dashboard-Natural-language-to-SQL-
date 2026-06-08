# 🤖 AI-Powered HR Analytics Dashboard

> **Convert Natural Language to SQL-Powered HR Insights in Real-Time**

## 📖 Overview

The **AI-Powered HR Analytics Dashboard** enables natural language interaction with HR data. Instead of writing complex SQL queries, users simply ask questions in plain English, and the system instantly delivers business-ready insights.

### 🎯 Key Features

- 🗣️ **Natural Language Processing** - Ask questions in plain English
- 🔄 **AI-to-SQL Translation** - Automatically converts questions into SQL queries
- 📊 **Interactive Dashboard** - Real-time visualization of HR metrics
- ⚡ **Instant Insights** - Get answers in seconds
- 🎨 **Beautiful UI** - Modern Streamlit-based interface with KPI cards
- 🔐 **Data Security** - Secure database connections

## 🚀 Tech Stack

- **Frontend**: Streamlit
- **AI/LLM**: Groq LLaMA 3.1 8B
- **Database**: Microsoft SQL Server
- **Language**: Python
- **Key Libraries**: 
  - `streamlit` - UI Framework
  - `pyodbc` - SQL Server connection
  - `pandas` - Data processing
  - `groq` - LLM integration

## 📋 Project Structure

```
├── mian_code.py          # Main application code (Streamlit Dashboard)
├── project report.docx   # Project documentation
├── README.md             # This file
└── .gitignore           # Git ignore rules
```

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.8+
- Microsoft SQL Server with HR data
- Groq API Key
- ODBC Driver 17 for SQL Server

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sujaychoukhande-06/AI-powered-HR-Analytics-AI-Dashboard-Natural-language-to-SQL-.git
   cd AI-powered-HR-Analytics-AI-Dashboard-Natural-language-to-SQL-
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install streamlit pyodbc pandas groq
   ```

4. **Set up environment variables**
   ```bash
   # Create a .env file or export:
   export GROQ_API_KEY="your-groq-api-key"
   ```

5. **Configure database** (Update in mian_code.py)
   ```python
   DATABASE_NAME = "HR_DB"        # Your database name
   SERVER_NAME = "your-server"    # Your SQL Server
   TABLE_NAME = "dbo.HR_EMP"     # Your HR employee table
   ```

6. **Run the application**
   ```bash
   streamlit run mian_code.py
   ```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 How It Works

### User Interaction Flow

```
User Question (Natural Language)
         ↓
NLP Processing & Intent Extraction
         ↓
AI-Powered SQL Query Generation
         ↓
Database Query Execution
         ↓
Results Display & AI Explanation
```

### Example Queries

```
"What is the average salary by department?"
"Show me employees with high performance ratings"
"List all employees in the Engineering department"
"Who are the below-average earning roles with high employee count?"
```

## 🎨 Dashboard Features

### KPI Cards
- 📈 **Total Employees** - Overall workforce count
- 👤 **Average Age** - Mean age of workforce
- 👨 **Male Employees** - Count of male employees
- 👩 **Female Employees** - Count of female employees

### HR Filters (Sidebar)
- Gender filter
- Department filter
- Job Role filter
- Education Field filter

All filters are dynamic and affect KPIs, results, and insights in real-time.

### Analysis Output
- 📋 **Query Results Table** - Displayed in interactive format
- 🧠 **Executive AI Insight** - AI-generated business explanation
- 🔍 **Generated SQL** - View the SQL query used

## 🔧 Configuration

Edit `mian_code.py` to configure:

```python
# Database Configuration
DATABASE_NAME = "HR_DB"
SERVER_NAME = "LAPTOP-NF72MPNC"
TABLE_NAME = "dbo.HR_EMP"

# Groq API
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
```

## 📝 Key Components in Code

### Main Functions

- **`extract_intent()`** - Uses Groq LLM to convert questions to structured intents
- **`build_sql()`** - Converts intent to SQL query
- **`execute_query()`** - Executes query and returns results
- **`explain_result()`** - Generates AI explanation for results
- **`normalize_question()`** - Normalizes user questions with semantic mapping

### Special Features

- **Question Normalization** - Maps common HR terms to database columns
- **Intent Detection** - Identifies query patterns (aggregations, filters, grouping)
- **Special Query Handler** - Handles complex queries like "below-average high-count roles"
- **Filter Integration** - All filters apply across KPIs and results

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 👨‍💼 Author

**Sujay Choukhande**
- GitHub: [@sujaychoukhande-06](https://github.com/sujaychoukhande-06)

## 🙏 Acknowledgments

- Groq for LLaMA 3.1 models
- Streamlit for the amazing UI framework
- Microsoft SQL Server documentation

---

<div align="center">
  Made with ❤️ for HR Analytics
</div>
