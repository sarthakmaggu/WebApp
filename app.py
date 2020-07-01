import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title("Binary Classsification Web App")
    st.sidebar.title("Binary Classification")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous ? 🍄")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('Mushroom.csv')
        label = LabelEncoder()
        standar = StandardScaler()
        for col in data.columns:
            data[col] = label.fit_transform(data[col].values.reshape(-1,1))
            if(col != "class"):
                data[col] = standar.fit_transform(data[col].values.reshape(-1,1))
        return data
    
    @st.cache(persist = True)
    def train_test(data):
        y = data["class"]
        x = data.drop(columns = ["class"])
        train_x,test_x,train_y,test_y = train_test_split(x,y, test_size = 0.3, random_state = 50)
        return train_x,test_x,train_y,test_y
    
    def plot(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model,test_x,test_y,display_labels=class_names)
            st.pyplot()
        
        if "ROC_Curve" in metrics_list:
            st.subheader("ROC Curve")
            plot_precision_recall_curve(model,test_x,test_y)
            st.pyplot()
        
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision Recall")
            plot_precision_recall_curve(model,test_x,test_y)
            st.pyplot()

    data = load_data()
    train_x,test_x,train_y,test_y = train_test(data)
    class_names = ["edible","not edible"]
    st.sidebar.subheader("Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C", 0.01, 10.0, step= 0.01, key ="C")
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key = "kernel")
        gamma = st.sidebar.radio("Gamma", ("scale", "auto"), key = "gamma")
        metrics = st.sidebar.multiselect("Metrics", ("Confusion Matrix", "ROC_Curve", "Precision-Recall Curve"))
        
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("SVM Results")
            model = SVC(C=C, kernel=kernel,gamma=gamma)
            model.fit(train_x,train_y)
            y_pred = model.predict(test_x)
            accuracy = model.score(test_x,test_y)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(test_y, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(test_y, y_pred, labels=class_names).round(2))
            plot(metrics)
        
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C", 0.01, 10.0, step= 0.01, key ="C")
        max_iter = st.sidebar.slider("Maximum Iterations", 100, 500, key = "max_iter")
        metrics = st.sidebar.multiselect("Metrics", ("Confusion Matrix", "ROC_Curve", "Precision-Recall Curve"))
        
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter= max_iter)
            model.fit(train_x,train_y)
            y_pred = model.predict(test_x)
            accuracy = model.score(test_x,test_y)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(test_y, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(test_y, y_pred, labels=class_names).round(2))
            plot(metrics)
    
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of tress", 100, 500, step= 10, key ="n_estimators")
        max_depth = st.sidebar.number_input("Depth of the tree",1, 20, step = 1, key = "max_depth")
        bootstrap = st.sidebar.radio("Bootstrap", ("True", "False"), key = "bootstrap")
        metrics = st.sidebar.multiselect("Metrics", ("Confusion Matrix", "ROC_Curve", "Precision-Recall Curve"))
        
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth= max_depth, bootstrap= bootstrap)
            model.fit(train_x,train_y)
            y_pred = model.predict(test_x)
            accuracy = model.score(test_x,test_y)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(test_y, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(test_y, y_pred, labels=class_names).round(2))
            plot(metrics)



    if st.sidebar.checkbox("Dataset", False):
        st.subheader("Dataset (Classification)")
        st.write(data)
        st.markdown("The data has been preprocessed")
    
    
if __name__ == '__main__':
    main()


