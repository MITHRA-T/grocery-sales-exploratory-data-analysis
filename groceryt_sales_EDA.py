# STEP 1: Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# STEP 2: Create sample grocery dataset

np.random.seed(42)

n = 360

products = [
    "Rice", "Milk", "Bread", "Eggs",
    "Apples", "Cooking Oil", "Sugar", "Tea"
]

category = {
    "Rice": "Staples",
    "Milk": "Dairy",
    "Bread": "Bakery",
    "Eggs": "Dairy",
    "Apples": "Fruits",
    "Cooking Oil": "Staples",
    "Sugar": "Staples",
    "Tea": "Beverages"
}

price = {
    "Rice": 65,
    "Milk": 30,
    "Bread": 40,
    "Eggs": 8,
    "Apples": 120,
    "Cooking Oil": 150,
    "Sugar": 48,
    "Tea": 180
}

dates = pd.date_range("2025-01-01", periods=n)

product = np.random.choice(products, n)
quantity = np.random.randint(1, 9, n)

unit_price = [price[p] for p in product]

sales = np.array(unit_price) * quantity
sales = np.round(sales * np.random.uniform(0.9, 1.1, n), 2)

df = pd.DataFrame({
    "Date": dates,
    "Product": product,
    "Category": [category[p] for p in product],
    "Quantity": quantity,
    "Unit_Price": unit_price,
    "Sales": sales,
    "Region": np.random.choice(
        ["North", "South", "East", "West"], n
    ),
    "Channel": np.random.choice(
        ["Online", "In-store"], n
    )
})

# Add missing values for cleaning demonstration
df.loc[5:7, "Region"] = np.nan
df.loc[10:12, "Quantity"] = np.nan
df.loc[15:17, "Sales"] = np.nan

# STEP 3: Understand the dataset

print("DATASET INFORMATION")
print("-------------------")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
display(df.head())

print("\nDataset Description:")
display(df.describe())

print("\nMissing Values:")
display(df.isnull().sum())

# STEP 4: Data Cleaning

clean_df = df.copy()

clean_df["Region"] = clean_df["Region"].fillna("Unknown")

clean_df["Quantity"] = clean_df["Quantity"].fillna(
    clean_df["Quantity"].median()
)

clean_df["Sales"] = clean_df["Sales"].fillna(
    clean_df["Sales"].median()
)

clean_df["Month"] = clean_df["Date"].dt.month

print("\nMissing Values After Cleaning:")
display(clean_df.isnull().sum())

# STEP 5: Exploratory Data Analysis

total_sales = clean_df["Sales"].sum()
total_records = len(clean_df)
average_sales = clean_df["Sales"].mean()

product_sales = clean_df.groupby(
    "Product"
)["Sales"].sum().sort_values(ascending=False)

category_sales = clean_df.groupby(
    "Category"
)["Sales"].sum().sort_values(ascending=False)

monthly_sales = clean_df.groupby(
    "Month"
)["Sales"].sum().sort_index()

region_sales = clean_df.groupby(
    "Region"
)["Sales"].sum().sort_values(ascending=False)

print("\nKEY FINDINGS")
print("------------")

print("Total Sales: ₹", round(total_sales, 2))
print("Total Records:", total_records)
print("Average Sales: ₹", round(average_sales, 2))
print("Top Product:", product_sales.idxmax())
print("Top Category:", category_sales.idxmax())
print("Top Region:", region_sales.idxmax())
print("Highest Sales Month:", monthly_sales.idxmax())

# STEP 6: Create Professional Dashboard

fig = plt.figure(figsize=(16, 10), facecolor="#F3F6FB")

fig.suptitle(
    "GROCERY SALES ANALYTICS DASHBOARD",
    fontsize=22,
    fontweight="bold",
    color="#172B4D",
    y=0.97
)

fig.text(
    0.5, 0.925,
    "EXPLORATORY DATA ANALYSIS | PYTHON PROJECT",
    ha="center",
    fontsize=11,
    color="#64748B"
)

# KPI Cards

kpis = [
    ("TOTAL SALES", f"₹{total_sales:,.0f}", "#2563EB"),
    ("TOTAL RECORDS", str(total_records), "#0F766E"),
    ("AVERAGE SALES", f"₹{average_sales:,.0f}", "#7C3AED"),
    ("TOP PRODUCT", product_sales.idxmax(), "#D97706")
]

for i, (title, value, color) in enumerate(kpis):

    ax = fig.add_axes([0.055 + i * 0.23, 0.78, 0.205, 0.11])

    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_color("#DCE3EE")

    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(
        0.08, 0.68, title,
        transform=ax.transAxes,
        fontsize=10,
        color="#64748B",
        fontweight="bold"
    )

    ax.text(
        0.08, 0.22, value,
        transform=ax.transAxes,
        fontsize=19,
        color=color,
        fontweight="bold"
    )

# CHART 1: Product Sales Bar Chart

ax1 = fig.add_axes([0.07, 0.43, 0.40, 0.27])

product_sales.sort_values().plot(
    kind="barh",
    ax=ax1,
    color="#3B82F6"
)

ax1.set_title("Sales by Product", fontsize=14, fontweight="bold")
ax1.set_xlabel("Sales (₹)")
ax1.grid(axis="x", alpha=0.2)

# CHART 2: Monthly Sales Line Chart

ax2 = fig.add_axes([0.55, 0.43, 0.40, 0.27])

months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

ax2.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o",
    linewidth=2.5,
    color="#0F9D8A"
)

ax2.set_xticks(range(1, 13), months, rotation=25)
ax2.set_title("Monthly Sales Trend", fontsize=14, fontweight="bold")
ax2.set_ylabel("Sales (₹)")
ax2.grid(alpha=0.2)

# CHART 3: Category Donut Chart

ax3 = fig.add_axes([0.07, 0.08, 0.40, 0.27])

ax3.pie(
    category_sales.values,
    labels=category_sales.index,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={
        "width": 0.42,
        "edgecolor": "white"
    }
)

ax3.text(
    0, 0,
    f"₹{total_sales/1000:.1f}K",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold"
)

ax3.set_title(
    "Sales by Category",
    fontsize=14,
    fontweight="bold"
)

# CHART 4: Region Sales Bar Chart

ax4 = fig.add_axes([0.55, 0.08, 0.40, 0.27])

region_sales.plot(
    kind="bar",
    ax=ax4,
    color=["#2563EB", "#0F9D8A", "#8B5CF6", "#F59E0B"]
)

ax4.set_title("Sales by Region", fontsize=14, fontweight="bold")
ax4.set_ylabel("Sales (₹)")
ax4.tick_params(axis="x", rotation=0)
ax4.grid(axis="y", alpha=0.2)

# STEP 7: Save Dashboard PNG

plt.savefig(
    "grocery_sales_eda_dashboard.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

plt.show()

# STEP 8: Save Cleaned Dataset

clean_df.to_csv(
    "cleaned_grocery_sales.csv",
    index=False
)

print("\nPROJECT FILES CREATED SUCCESSFULLY!")

# STEP 9: Download Files

files.download("grocery_sales_eda_dashboard.png")

files.download("cleaned_grocery_sales.csv")