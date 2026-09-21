"""
LifeDash International Healthcare - Data Analytics Project
Run this file in PyCharm.

Dataset: LifeDash_International_Healthcare_Dataset_50K.xlsx
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "LifeDash_International_Healthcare_Dataset_50K.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


df = pd.read_excel(DATA_FILE)

print("\n========== DATASET OVERVIEW ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nColumns:")
print(df.columns.tolist())

df["Visit_Date"] = pd.to_datetime(df["Visit_Date"], errors="coerce")

duplicates = df.duplicated().sum()
df = df.drop_duplicates().copy()


missing_before = df.isna().sum().sum()
df = df.dropna().copy()
missing_after = df.isna().sum().sum()

print("\n========== DATA CLEANING ==========")
print("Duplicate rows found:", duplicates)
print("Missing values before cleaning:", missing_before)
print("Missing values after cleaning:", missing_after)

print("\n========== NUMERICAL SUMMARY ==========")
print(df[["Age", "Waiting_Time_Min", "Billing_Amount_USD"]].describe())

print("\n========== CATEGORICAL SUMMARY ==========")
for col in ["Country", "City", "Hospital_Branch", "Department",
            "Gender", "Patient_Status"]:
    print(f"\n{col}:")
    print(df[col].value_counts())

total_patients = len(df)
total_billing = df["Billing_Amount_USD"].sum()
average_bill = df["Billing_Amount_USD"].mean()
average_wait = df["Waiting_Time_Min"].mean()
median_wait = df["Waiting_Time_Min"].median()
average_age = df["Age"].mean()

print("\n========== KEY KPIs ==========")
print(f"Total Patients: {total_patients:,}")
print(f"Total Billing: ${total_billing:,.2f}")
print(f"Average Billing per Patient: ${average_bill:,.2f}")
print(f"Average Waiting Time: {average_wait:.2f} minutes")
print(f"Median Waiting Time: {median_wait:.2f} minutes")
print(f"Average Patient Age: {average_age:.2f} years")


country_patients = df["Country"].value_counts()
country_billing = df.groupby("Country")["Billing_Amount_USD"].sum().sort_values(ascending=False)

print("\n========== COUNTRY ANALYSIS ==========")
print("\nPatients by country:")
print(country_patients)
print("\nBilling by country:")
print(country_billing)

plt.figure(figsize=(10, 6))
country_patients.sort_values().plot(kind="barh")
plt.title("Number of Patients by Country")
plt.xlabel("Number of Patients")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_patients_by_country.png"), dpi=200)
plt.close()


dept_stats = df.groupby("Department").agg(
    Patients=("Patient_ID", "count"),
    Average_Wait=("Waiting_Time_Min", "mean"),
    Total_Billing=("Billing_Amount_USD", "sum"),
    Average_Billing=("Billing_Amount_USD", "mean")
).sort_values("Patients", ascending=False)

print("\n========== DEPARTMENT ANALYSIS ==========")
print(dept_stats)

plt.figure(figsize=(10, 6))
dept_stats["Patients"].sort_values().plot(kind="barh")
plt.title("Patient Volume by Department")
plt.xlabel("Number of Patients")
plt.ylabel("Department")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_patients_by_department.png"), dpi=200)
plt.close()

plt.figure(figsize=(10, 6))
dept_stats["Average_Wait"].sort_values().plot(kind="barh")
plt.title("Average Waiting Time by Department")
plt.xlabel("Average Waiting Time (minutes)")
plt.ylabel("Department")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_waiting_time_by_department.png"), dpi=200)
plt.close()


status_counts = df["Patient_Status"].value_counts()

print("\n========== PATIENT STATUS ==========")
print(status_counts)

plt.figure(figsize=(8, 6))
status_counts.plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.title("Patient Status Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_patient_status.png"), dpi=200)
plt.close()

gender_counts = df["Gender"].value_counts()

print("\n========== GENDER ==========")
print(gender_counts)

plt.figure(figsize=(8, 5))
gender_counts.plot(kind="bar")
plt.title("Patient Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_gender_distribution.png"), dpi=200)
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Patient Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_age_distribution.png"), dpi=200)
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(df["Billing_Amount_USD"], bins=30, kde=True)
plt.title("Billing Amount Distribution")
plt.xlabel("Billing Amount (USD)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_billing_distribution.png"), dpi=200)
plt.close()


plt.figure(figsize=(10, 6))
sns.histplot(df["Waiting_Time_Min"], bins=30, kde=True)
plt.title("Waiting Time Distribution")
plt.xlabel("Waiting Time (minutes)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_waiting_time_distribution.png"), dpi=200)
plt.close()



df["Month"] = df["Visit_Date"].dt.to_period("M").astype(str)
monthly = df.groupby("Month").agg(
    Patients=("Patient_ID", "count"),
    Billing=("Billing_Amount_USD", "sum")
)

print("\n========== MONTHLY TREND ==========")
print(monthly)

plt.figure(figsize=(12, 6))
plt.plot(monthly.index, monthly["Patients"], marker="o")
plt.title("Monthly Patient Visits")
plt.xlabel("Month")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_monthly_patient_trend.png"), dpi=200)
plt.close()


branch_stats = df.groupby("Hospital_Branch").agg(
    Patients=("Patient_ID", "count"),
    Average_Wait=("Waiting_Time_Min", "mean"),
    Total_Billing=("Billing_Amount_USD", "sum")
).sort_values("Patients", ascending=False)

print("\n========== HOSPITAL BRANCH ANALYSIS ==========")
print(branch_stats)


numeric_cols = ["Age", "Waiting_Time_Min", "Billing_Amount_USD"]
corr = df[numeric_cols].corr()

print("\n========== CORRELATION MATRIX ==========")
print(corr)

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_correlation_matrix.png"), dpi=200)
plt.close()

with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "analysis_results.xlsx")) as writer:
    df.describe(include="all").to_excel(writer, sheet_name="Summary")
    country_patients.rename("Patients").to_excel(writer, sheet_name="Country_Patients")
    country_billing.rename("Billing").to_excel(writer, sheet_name="Country_Billing")
    dept_stats.to_excel(writer, sheet_name="Departments")
    status_counts.rename("Patients").to_excel(writer, sheet_name="Status")
    branch_stats.to_excel(writer, sheet_name="Branches")
    monthly.to_excel(writer, sheet_name="Monthly_Trend")
    corr.to_excel(writer, sheet_name="Correlation")


top_country = country_patients.idxmax()
top_department = dept_stats["Patients"].idxmax()
highest_wait_department = dept_stats["Average_Wait"].idxmax()
highest_billing_department = dept_stats["Total_Billing"].idxmax()
top_status = status_counts.idxmax()
busiest_date = df["Visit_Date"].value_counts().idxmax()

print("\n========== KEY INSIGHTS ==========")
print(f"1. {top_country} has the highest number of patient records.")
print(f"2. {top_department} has the highest patient volume.")
print(f"3. {highest_wait_department} has the highest average waiting time.")
print(f"4. {highest_billing_department} generates the highest total billing.")
print(f"5. {top_status} is the most common patient status.")
print(f"6. The busiest recorded date is {busiest_date.date()}.")

print("\nProject completed successfully.")
print("Charts and Excel analysis tables are available in the 'outputs' folder.")
