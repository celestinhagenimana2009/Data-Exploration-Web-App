import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simple EDA App", layout="wide")

st.title("Simple EDA App")
st.write("Upload any CSV file and explore its summary, missing values, and plots.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success("File uploaded successfully.")

    st.subheader("1. Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("2. Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("3. Column Information")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Missing %": (df.isnull().mean() * 100).round(2).values,
        "Unique Values": df.nunique().values
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("4. Missing Values")
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column", "Missing Count"]
    st.dataframe(missing_df, use_container_width=True)

    st.subheader("5. Summary Statistics")
    st.write("Numerical columns:")
    st.dataframe(df.describe().T, use_container_width=True)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cat_cols) > 0:
        st.write("Categorical columns:")
        st.dataframe(df[cat_cols].describe().T, use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("6. Univariate Plots")

    if len(numeric_cols) > 0:
        selected_num_col = st.selectbox("Choose a numerical column for histogram", numeric_cols)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[selected_num_col].dropna(), bins=20, edgecolor="black")
        ax.set_title(f"Histogram of {selected_num_col}")
        ax.set_xlabel(selected_num_col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 2))
        ax2.boxplot(df[selected_num_col].dropna(), vert=False)
        ax2.set_title(f"Boxplot of {selected_num_col}")
        st.pyplot(fig2)
    else:
        st.info("No numerical columns found for histogram and boxplot.")

    st.subheader("7. Categorical Value Counts")
    if len(cat_cols) > 0:
        selected_cat_col = st.selectbox("Choose a categorical column", cat_cols)

        value_counts = df[selected_cat_col].astype(str).value_counts().reset_index()
        value_counts.columns = [selected_cat_col, "Count"]
        st.dataframe(value_counts, use_container_width=True)

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        value_counts.head(10).plot(
            kind="bar",
            x=selected_cat_col,
            y="Count",
            ax=ax3,
            legend=False
        )
        ax3.set_title(f"Top 10 Categories in {selected_cat_col}")
        ax3.set_xlabel(selected_cat_col)
        ax3.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig3)
    else:
        st.info("No categorical columns found.")

    st.subheader("8. Correlation Heatmap")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()

        fig4, ax4 = plt.subplots(figsize=(8, 6))
        cax = ax4.imshow(corr, interpolation="nearest", aspect="auto")
        ax4.set_title("Correlation Heatmap")
        ax4.set_xticks(range(len(corr.columns)))
        ax4.set_yticks(range(len(corr.columns)))
        ax4.set_xticklabels(corr.columns, rotation=90)
        ax4.set_yticklabels(corr.columns)
        fig4.colorbar(cax)
        st.pyplot(fig4)

        st.dataframe(corr, use_container_width=True)
    else:
        st.info("Need at least 2 numerical columns for correlation heatmap.")

    st.subheader("9. Download Clean Summary")
    summary_text = []
    summary_text.append(f"Rows: {df.shape[0]}")
    summary_text.append(f"Columns: {df.shape[1]}")
    summary_text.append("\nColumn Information:\n")
    summary_text.append(info_df.to_string(index=False))
    summary_text.append("\n\nNumerical Summary:\n")
    summary_text.append(df.describe().to_string())

    summary_output = "\n".join(summary_text)

    st.download_button(
        label="Download Summary as TXT",
        data=summary_output,
        file_name="eda_summary.txt",
        mime="text/plain"
    )

else:
    st.info("Please upload a CSV file to begin.")