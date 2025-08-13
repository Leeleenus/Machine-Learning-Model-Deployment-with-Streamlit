import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset.csv") 

model = load_model()
df = load_data()


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Visualizations", "Model Prediction", "Model Performance"])


if page == "Data Exploration":
    st.title("Data Exploration")
    st.write("### Dataset Overview")
    st.write(df.shape)
    st.write(df.dtypes)
    st.write(df.head())

    st.write("### Filter Data")
    columns = st.multiselect("Select columns to display", df.columns.tolist(), default=df.columns.tolist())
    st.dataframe(df[columns])


elif page == "Visualizations":
    st.title("Visualizations")


    st.subheader("Histogram")
    col = st.selectbox("Select column for histogram", df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)


    st.subheader("Scatter Plot")
    x_axis = st.selectbox("X-axis", df.columns)
    y_axis = st.selectbox("Y-axis", df.columns)
    fig2 = px.scatter(df, x=x_axis, y=y_axis, color=df.columns[-1])
    st.plotly_chart(fig2)


    st.subheader("Box Plot")
    box_col = st.selectbox("Select column for box plot", df.columns)
    fig3, ax3 = plt.subplots()
    sns.boxplot(x=df[box_col], ax=ax3)
    st.pyplot(fig3)


elif page == "Model Prediction":
    st.title("🔮 Model Prediction")

    st.write("Enter feature values to get a prediction:")

    inputs = []
    for col in df.columns[:-1]:  
        val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        inputs.append(val)

    if st.button("Predict"):
        prediction = model.predict([inputs])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([inputs]).max()
            st.success(f"Prediction: {prediction[0]} (Confidence: {proba:.2f})")
        else:
            st.success(f"Prediction: {prediction[0]}")


elif page == "Model Performance":
    from sklearn.metrics import classification_report, confusion_matrix
    st.title("📏 Model Performance")

    X = df.drop(columns=[df.columns[-1]])
    y = df[df.columns[-1]]
    y_pred = model.predict(X)

    st.text("Classification Report:")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig4, ax4 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4)
    st.pyplot(fig4)
