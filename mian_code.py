import streamlit as st
import os
import pyodbc
import pandas as pd
import json
from groq import Groq

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if "question_text" not in st.session_state:
    st.session_state.question_text = ""

if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

# =====================================================
# CONFIG
# =====================================================
DATABASE_NAME = "HR_DB"
SERVER_NAME = "LAPTOP-NF72MPNC"
TABLE_NAME = "dbo.HR_EMP"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not set")
    st.stop()

# =====================================================
# SQL CONNECTION
# =====================================================
conn = pyodbc.connect(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SERVER_NAME};"
    f"DATABASE={DATABASE_NAME};"
    "Trusted_Connection=yes;"
)

# =====================================================
# SIDEBAR HR FILTERS (ADDED)
# =====================================================
filter_df = pd.read_sql(f"""
SELECT DISTINCT Gender, Department, JobRole, EducationField
FROM {TABLE_NAME}
""", conn)

st.sidebar.markdown("""
<div style="
    background:linear-gradient(135deg,#1e3a8a,#2563eb);
    padding:16px;
    border-radius:12px;
    margin-bottom:15px;
">
    <h3 style="color:white;margin:0;">🔎 HR Filters</h3>
</div>
""", unsafe_allow_html=True)

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=sorted(filter_df["Gender"].dropna().unique()),
    default=[]
)

department_filter = st.sidebar.multiselect(
    "Department",
    options=sorted(filter_df["Department"].dropna().unique()),
    default=[]
)

jobrole_filter = st.sidebar.multiselect(
    "Job Role",
    options=sorted(filter_df["JobRole"].dropna().unique()),
    default=[]
)

education_filter = st.sidebar.multiselect(
    "Education Field",
    options=sorted(filter_df["EducationField"].dropna().unique()),
    default=[]
)

st.sidebar.caption("Filters affect KPIs, table, and insights")


# =====================================================
# KPI DATA
# =====================================================
kpi_df = pd.read_sql(f"""
SELECT
    COUNT(*) AS TotalEmployees,
    AVG(Age) AS AvgAge,
    SUM(CASE WHEN Gender='Male' THEN 1 ELSE 0 END) AS MaleEmployees,
    SUM(CASE WHEN Gender='Female' THEN 1 ELSE 0 END) AS FemaleEmployees
FROM {TABLE_NAME};
""", conn)

# =====================================================
# COLUMN METADATA
# =====================================================
schema_df = pd.read_sql(
    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='HR_EMP'",
    conn
)
VALID_COLUMNS = set(schema_df["COLUMN_NAME"].tolist())

COLUMN_METADATA = {
    "Age": "numeric",
    "Attrition": "categorical",
    "BusinessTravel": "categorical",
    "DailyRate": "numeric",
    "Department": "categorical",
    "DistanceFromHome": "numeric",
    "Education": "numeric",
    "EducationField": "categorical",
    "EmployeeCount": "numeric",
    "EmployeeNumber": "numeric",
    "EnvironmentSatisfaction": "numeric",
    "Gender": "categorical",
    "HourlyRate": "numeric",
    "JobInvolvement": "numeric",
    "JobLevel": "numeric",
    "JobRole": "categorical",
    "JobSatisfaction": "numeric",
    "MaritalStatus": "categorical",
    "MonthlyIncome": "numeric",
    "MonthlyRate": "numeric",
    "NumCompaniesWorked": "numeric",
    "Over18": "categorical",
    "OverTime": "categorical",
    "PercentSalaryHike": "numeric",
    "PerformanceRating": "numeric",
    "RelationshipSatisfaction": "numeric",
    "StandardHours": "numeric",
    "StockOptionLevel": "numeric",
    "TotalWorkingYears": "numeric",
    "TrainingTimesLastYear": "numeric",
    "WorkLifeBalance": "numeric",
    "YearsAtCompany": "numeric",
    "YearsInCurrentRole": "numeric",
    "YearsSinceLastPromotion": "numeric",
    "YearsWithCurrManager": "numeric"
}

# =====================================================
# SPECIAL QUERY DETECTOR
# =====================================================
def is_below_avg_high_count_query(question: str) -> bool:
    q = question.lower()
    return (
        ("below-average" in q or "below average" in q)
        and ("high employee" in q or "high employee count" in q)
        and ("job role" in q or "job roles" in q)
    )

# =====================================================
# SPECIAL SQL
# =====================================================
def build_below_avg_high_count_sql() -> str:
    return f"""
    WITH overall_income AS (
        SELECT AVG(MonthlyIncome) AS overall_avg_income FROM {TABLE_NAME}
    ),
    role_stats AS (
        SELECT JobRole,
               COUNT(*) AS employee_count,
               AVG(MonthlyIncome) AS avg_income
        FROM {TABLE_NAME}
        GROUP BY JobRole
    ),
    benchmark AS (
        SELECT AVG(employee_count) AS avg_employee_count FROM role_stats
    )
    SELECT r.JobRole, r.employee_count, r.avg_income
    FROM role_stats r
    CROSS JOIN overall_income i
    CROSS JOIN benchmark b
    WHERE r.avg_income < i.overall_avg_income
      AND r.employee_count > b.avg_employee_count
    ORDER BY r.employee_count DESC;
    """

# =====================================================
# USER LANGUAGE NORMALIZATION
# =====================================================
SEMANTIC_MAP = {
    "salary": "MonthlyIncome",
    "income": "MonthlyIncome",
    "pay": "MonthlyIncome",
    "overtime": "OverTime",
    "travel frequently": "Travel_Frequently",
    "travel rarely": "Travel_Rarely",
    "no travel": "Non-Travel",
    "education field": "EducationField",
    "job role": "JobRole",
    "experience": "TotalWorkingYears",
    "promotion": "YearsSinceLastPromotion",
    "manager": "YearsWithCurrManager",
    "work life balance": "WorkLifeBalance",
    "performance": "PerformanceRating"
}

def normalize_question(question: str) -> str:
    q = question.lower()
    for k, v in SEMANTIC_MAP.items():
        q = q.replace(k, v)
    return q

# =====================================================
# INTENT EXTRACTION
# =====================================================
def extract_intent(question: str) -> dict:
    prompt = f"""
Return STRICT JSON only.
No explanation.
No markdown.

Rules:
- Use ONLY these columns:
{", ".join(COLUMN_METADATA.keys())}

- Create filters when user specifies conditions
- For categorical columns use '='
- For numeric columns allow comparisons
- String values should NOT have quotes

Format:
{{
  "metrics":[{{"type":"count","column":"*"}}],
  "group_by":[],
  "filters":[]
}}

Question:
{question}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = res.choices[0].message.content.strip()

    try:
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()
        intent = json.loads(raw)
    except Exception:
        intent = {
            "metrics": [{"type": "count"}],
            "group_by": [],
            "filters": []
        }

    intent["group_by"] = [
        g["column"] if isinstance(g, dict) else g
        for g in intent.get("group_by", [])
    ]
    return intent

# =====================================================
# SQL VALUE FORMATTER
# =====================================================
def sql_value(value, column):
    if COLUMN_METADATA.get(column) == "categorical":
        return f"'{value}'"
    return value

# =====================================================
# SQL BUILDER
# =====================================================
def build_sql(intent: dict) -> str:
    select_parts = []
    for m in intent["metrics"]:
        col = m.get("column", "*")
        func = m["type"].upper()
        alias = f"{m['type']}_{'all' if col == '*' else col}"
        select_parts.append(f"{func}({col}) AS {alias}")

    group_by = intent.get("group_by", [])
    sql = f"SELECT {', '.join(select_parts + group_by)} FROM {TABLE_NAME}"

   
    # ✅ ALWAYS initialize conditions
    conditions = []

    for f in intent.get("filters", []):
        column = f.get("column")
        value = f.get("value")

        # Default operator if missing
        operator = f.get("operator", "=")

        if column and value is not None:
            conditions.append(
                f"{column} {operator} {sql_value(value, column)}"
            )

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    if group_by:
        sql += " GROUP BY " + ", ".join(group_by)

    return sql + ";"



def is_list_query(question: str) -> bool:
    q = question.lower()
    return any(
        phrase in q
        for phrase in [
            "list of employees",
            "show employees",
            "give me employees",
            "employee list",
            "employee details"
        ]
    )


# =====================================================
# EXECUTE QUERY
# =====================================================
def execute_query(question: str):
    question = normalize_question(question)

    if is_list_query(question):
        sql = f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE Age > 39;
        """
    elif is_below_avg_high_count_query(question):
        sql = build_below_avg_high_count_sql()
    else:
        intent = extract_intent(question)
        sql = build_sql(intent)

    df = pd.read_sql(sql, conn)
    return df, sql


# =====================================================
# AI EXPLANATION
# =====================================================

def explain_result(question: str, df: pd.DataFrame) -> str:
    total_rows = len(df)

    preview = df.head(10).to_dict(orient="records")

    prompt = f"""
You are an HR analytics expert.

IMPORTANT RULES:
- Total number of records = {total_rows}
- The table below is ONLY A SAMPLE (first 10 rows)
- DO NOT calculate totals or percentages from the sample
- DO NOT say numbers like "7 out of 12"

User Question:
{question}

Sample Records:
{preview}

Explain insights in business language.
Focus on patterns and implications, not recalculating counts.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()

# =====================================================
# STREAMLIT UI (RESTORED AS PER IMAGE)
# =====================================================
st.set_page_config(page_title="HR Analytics", layout="wide")

st.markdown("""
<style>
.kpi-blue { background: linear-gradient(135deg,#2563eb,#1e40af); }
.kpi-purple { background: linear-gradient(135deg,#7c3aed,#4c1d95); }
.kpi-green { background: linear-gradient(135deg,#16a34a,#065f46); }
.kpi-pink { background: linear-gradient(135deg,#db2777,#831843); }

.kpi-card {
  border-radius: 18px;
  padding: 26px;
  text-align: center;
  color: white;
  box-shadow: 0 18px 40px rgba(0,0,0,0.18);
}

.kpi-title { font-size: 14px; font-weight: 600; }
.kpi-value { font-size: 38px; font-weight: 800; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 HR Analytics – Dashboard")

# =====================================================
# FILTERED MASTER DATA (FOR KPI & RESULTS)
# =====================================================
master_df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)

if gender_filter:
    master_df = master_df[master_df["Gender"].isin(gender_filter)]

if department_filter:
    master_df = master_df[master_df["Department"].isin(department_filter)]

if jobrole_filter:
    master_df = master_df[master_df["JobRole"].isin(jobrole_filter)]

if education_filter:
    master_df = master_df[master_df["EducationField"].isin(education_filter)]

# Override KPI values dynamically
total_emp = master_df.shape[0]
avg_age = int(round(master_df["Age"].mean())) if total_emp else 0
male_emp = master_df[master_df["Gender"] == "Male"].shape[0]
female_emp = master_df[master_df["Gender"] == "Female"].shape[0]


# ===============================
# FILTER-AWARE KPI CALCULATION
# ===============================
total_emp = master_df.shape[0]
avg_age = int(round(master_df["Age"].mean())) if total_emp else 0
male_emp = master_df[master_df["Gender"] == "Male"].shape[0]
female_emp = master_df[master_df["Gender"] == "Female"].shape[0]


st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:22px;margin-top:18px;">
  <div class="kpi-card kpi-blue"><div class="kpi-title">Total Employees</div><div class="kpi-value">{total_emp}</div></div>
  <div class="kpi-card kpi-purple"><div class="kpi-title">Average Age</div><div class="kpi-value">{avg_age}</div></div>
  <div class="kpi-card kpi-green"><div class="kpi-title">Male Employees</div><div class="kpi-value">{male_emp}</div></div>
  <div class="kpi-card kpi-pink"><div class="kpi-title">Female Employees</div><div class="kpi-value">{female_emp}</div></div>
</div>
""", unsafe_allow_html=True)

question = st.text_input("Ask a business question", value=st.session_state.question_text)

if st.button("▶ Run Analysis"):
    # 1️⃣ EXECUTE QUERY FIRST (df is CREATED HERE)
    df, sql = execute_query(question)

    st.success("Query executed successfully")

    # =====================================================
    # PART 3: QUERY RESULTS (df EXISTS NOW)
    # =====================================================
    st.subheader("📋 Query Results")

    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#1e3a8a,#2563eb);
        padding:14px;
        border-radius:10px;
        margin-bottom:8px;
    ">
        <h4 style="color:white;margin:0;">📊 Analytical Output Table</h4>
    </div>
    """, unsafe_allow_html=True)

    # ✅ NOW df IS DEFINED — SAFE TO USE
    filtered_df = df.copy()

    if "Gender" in filtered_df.columns and gender_filter:
        filtered_df = filtered_df[filtered_df["Gender"].isin(gender_filter)]

    if "Department" in filtered_df.columns and department_filter:
        filtered_df = filtered_df[filtered_df["Department"].isin(department_filter)]

    if "JobRole" in filtered_df.columns and jobrole_filter:
        filtered_df = filtered_df[filtered_df["JobRole"].isin(jobrole_filter)]

    if "EducationField" in filtered_df.columns and education_filter:
        filtered_df = filtered_df[filtered_df["EducationField"].isin(education_filter)]

    st.dataframe(filtered_df, use_container_width=True)

    # =====================================================
    # PART 4: EXECUTIVE AI INSIGHT (USE filtered_df)
    # =====================================================
    explanation = explain_result(question, filtered_df)

    st.markdown(f"""
    <div style="
        background:#F0F7FF;
        padding:20px;
        border-left:6px solid #2563EB;
        border-radius:8px;
        margin-top:18px;
    ">
        <h4>🧠 Executive AI Insight</h4>
        <p style="font-size:15px;">{explanation}</p>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # SQL VIEW
    # =====================================================
    with st.expander("🔍 Generated SQL"):
        st.code(sql, language="sql")

