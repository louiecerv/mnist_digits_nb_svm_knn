import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load MNIST dataset
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
nb_classifier = GaussianNB()
svm_classifier = SVC()
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train classifiers
nb_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)

# Predict
nb_predictions = nb_classifier.predict(X_test)
svm_predictions = svm_classifier.predict(X_test)
knn_predictions = knn_classifier.predict(X_test)

# Compute accuracy
nb_accuracy = accuracy_score(y_test, nb_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)

# Compute classification reports and confusion matrices
nb_report = classification_report(y_test, nb_predictions)
svm_report = classification_report(y_test, svm_predictions)
knn_report = classification_report(y_test, knn_predictions)

nb_cm = confusion_matrix(y_test, nb_predictions)
svm_cm = confusion_matrix(y_test, svm_predictions)
knn_cm = confusion_matrix(y_test, knn_predictions)

def main():
    # Streamlit App
    st.title("MNIST Classifier Performance")
    st.write("### Sample Images from MNIST Dataset")

    about = """# 🖥️ MNIST Classifier Performance App 🚀
This Streamlit app demonstrates the performance of three different machine learning classifiers on the **MNIST handwritten digits dataset**. 📊 The classifiers compared are:

✅ **Naïve Bayes**  
✅ **Support Vector Machine (SVM)**  
✅ **K-Nearest Neighbors (KNN)**  

## 🔍 Features:
- 📸 **Displays 5 sample images** from the MNIST dataset.
- 📊 **Trains and evaluates** Naïve Bayes, SVM, and KNN classifiers.
- 🏆 **Compares classifier accuracy** on the test dataset.
- 📄 **Shows classification reports** with precision, recall, and F1-score.
- 🔥 **Visualizes confusion matrices** using heatmaps for better understanding.

## 📌 How to Use:
1. Run the app using Streamlit.  
2. Navigate through the **three tabs** to check the performance of each classifier.  
3. Analyze the **classification report and confusion matrix** for deeper insights.  
4. Read the **comparison section** to understand the strengths and weaknesses of each model.  

## 🎯 Insights:
- **Naïve Bayes**: Fast but may struggle with complex patterns.  
- **SVM**: Balanced performance with good accuracy.  
- **KNN**: Effective but can be slow with large datasets.  

🚀 Explore and experiment with different models to enhance classification performance!  

### 📌 About the Creator  
**Created by:** *Louie F. Cervantes, M.Eng. (Information Engineering)*  
**(c) 2025 West Visayas State University**
"""
    with st.expander("About the App"):
        st.markdown(about)

    # Display 5 sample images
    fig, axes = plt.subplots(1, 5, figsize=(10, 3))
    for i, ax in enumerate(axes):
        ax.imshow(digits.images[i], cmap='gray')
        ax.set_title(f"Label: {digits.target[i]}")
        ax.axis('off')
    st.pyplot(fig)

    st.markdown("## Clcik on the tabs below to view classifier performance:")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Naïve Bayes", "SVM", "KNN"])

    with tab1:
        st.subheader("Naïve Bayes Classifier")
        st.write(f"Accuracy: {nb_accuracy:.4f}")
        st.write("Classification Report:")
        st.write(nb_report)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    with tab2:
        st.subheader("Support Vector Machine (SVM)")
        st.write(f"Accuracy: {svm_accuracy:.4f}")
        st.write("Classification Report:")
        st.write(svm_report)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    with tab3:
        st.subheader("K-Nearest Neighbors (KNN)")
        st.write(f"Accuracy: {knn_accuracy:.4f}")
        st.write("Classification Report:")
        st.write(knn_report)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    # Comparison
    st.write("## Classifier Comparison")
    st.write("### Observations:")
    st.write("- **Naïve Bayes** is fast but may struggle with complex patterns.")
    st.write("- **SVM** performs well with a balance of accuracy and speed.")
    st.write("- **KNN** can be effective but may be slower with large datasets.")

if __name__ == "__main__":
    main()