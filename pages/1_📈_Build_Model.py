import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import xgboost as xgb
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             r2_score, mean_squared_error, mean_absolute_error)
from sklearn.utils import resample
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pickle
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_class_distribution(y, title="Class Distribution"):

    class_counts = pd.Series(y).value_counts()
    
    class_labels = class_counts.index
    frequencies = class_counts.values

    class_distribution = pd.DataFrame({'Class': class_labels, 'Frequency': frequencies})

    st.subheader(title)
    st.bar_chart(class_distribution, x="Class", y="Frequency", horizontal=True)
    
def plot_and_interpret_learning_curve(model, X, y, cv):
    st.subheader("Learning Curve Interpretation")

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title("Learning Curve")
    plt.xlabel("Training Samples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    st.pyplot(plt)

    # Interpretation
    st.write(f'''
            Cross-validation score : {cv} \n
            Train scores mean : {train_scores_mean[-1]:.4f} \n
            Test scores mean : {test_scores_mean[-1]:.4f}
            ''')
    
    score_diff = train_scores_mean[-1] - test_scores_mean[-1]
    score_improvement = test_scores_mean[-1] - test_scores_mean[0]
    
    if test_scores_mean[-1] < 0.7: 
        st.warning("The model may be underfitting. Consider:")
        st.write("- Increasing model complexity")
        st.write("- Adding more relevant features")
        st.write("- Collecting more diverse training data")
    
    if score_diff > 0.2:  
        st.warning("The model may be overfitting. Consider:")
        st.write("- Using regularization")
        st.write("- Simplifying the model")
        st.write("- Collecting more training data")
    
    if score_improvement < 0.05:
        st.info("The learning curve shows limited improvement. Consider:")
        st.write("- Trying a different model architecture")
        st.write("- Feature engineering or selection")
    
    if abs(test_scores_mean[-1] - test_scores_mean[-2]) < 0.01:
        st.success("The learning curve appears to have converged.")
    else:
        st.info("The model might benefit from more training data.")

    if 0.7 <= test_scores_mean[-1] < 0.9 and score_diff <= 0.2:
        st.success("The model shows good fit to the data.")
    elif test_scores_mean[-1] >= 0.9:
        st.success("The model shows excellent performance!")
        
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis_r', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)
    
def plot_regression_results(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    st.pyplot(plt)

def calculate_regression_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = len(y_pred) 
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    st.write(f"R-squared: {r2:.4f}")
    st.write(f"Adjusted R-squared: {adjusted_r2:.4f}")
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"Root Mean Squared Error: {rmse:.4f}")
    st.write(f"Mean Absolute Error: {mae:.4f}")

def perform_hypothesis_test(y_true, y_pred):
    st.subheader("Hypothesis Testing")
    
    t_statistic, p_value = stats.ttest_rel(y_true, y_pred)
    
    st.write("Null Hypothesis: There is no significant difference between the true values and the predicted values.")
    st.write("Alternative Hypothesis: There is a significant difference between the true values and the predicted values.")
    st.write(f"T-statistic: {t_statistic:.4f}")
    st.write(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        st.success("Reject the null hypothesis: The model has significant predictive power.")
    else:
        st.info("Fail to reject the null hypothesis: The model does not have significant predictive power.")


def build_model():
    st.title("Build Machine Learning Model")
    
    # Upload data
    uploaded_file = st.file_uploader("Choose a CSV file for training", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.info("Data Preview")
        st.write(data.head())
        st.write("Rows: ", data.shape[0])

        st.write("---")
        # Select features and target
        st.subheader("Preprocess Data")
        features = st.multiselect("Select features", data.columns)
        target = st.selectbox("Select target variable", data.columns)

        if len(features) > 0 and target:
            
            if target in features:
                st.warning("Target variable should not be in features.")
                features.remove(target)

            
            selected_data = data[features + [target]]

            
            if st.checkbox("Drop NA values"):
                original_shape = selected_data.shape
                selected_data = selected_data.dropna()
                new_shape = selected_data.shape
                st.info(f"New Rows: {new_shape[0]}")
                
                st.download_button(
                        label="Download Clean Data",
                        data=selected_data.to_csv(index=False),
                        file_name="Clean data.csv",
                        mime="csv"
                    )
            
        
            X = selected_data[features]
            y = selected_data[target]
            
            st.info("Updated Data Preview")
            st.write(selected_data.head())
            
            st.write("---")
            
            # Encode target labels
            if st.checkbox("Encode target labels"):
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                
                
                st.subheader("Label Encoding")
                encoding_info = pd.DataFrame({
                    'Original': label_encoder.classes_,
                    'Encoded': range(len(label_encoder.classes_))
                })
                st.write(encoding_info)

            plot_class_distribution(y, title="Class Distribution")

            resampling_choice = st.selectbox("Choose resampling technique", 
                                             ["None", "Oversampling", "Undersampling", "SMOTE"])

            if resampling_choice == "Oversampling":
                ros = RandomOverSampler(random_state=42)
                X_resampled, y_resampled = ros.fit_resample(X, y)
                plot_class_distribution(y_resampled, title="Oversampled Class Distribution")
            elif resampling_choice == "Undersampling":
                rus = RandomUnderSampler(random_state=42)
                X_resampled, y_resampled = rus.fit_resample(X, y)
                plot_class_distribution(y_resampled, title="Undersampled Class Distribution")
            elif resampling_choice == "SMOTE":
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                plot_class_distribution(y_resampled, title="SMOTE Class Distribution")
            else:
                X_resampled, y_resampled = X, y

            st.write("---")
            
            st.subheader("Build Model")
            test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=test_size, random_state=42)

            problem_type = st.selectbox("Choose problem type", ["Classification", "Regression"])

            validation_method = "Holdout Set"

            # Choose model
            if problem_type == "Classification":
                model_choice = st.selectbox("Choose a model", 
                                            ["Random Forest", "Logistic Regression", "Support Vector Machine", 
                                             "Decision Tree", "K-Nearest Neighbors", "XGBoost"])
            else: 
                model_choice = st.selectbox("Choose a model", 
                                            ["Random Forest", "Linear Regression", "Support Vector Machine", 
                                              "K-Nearest Neighbors", "XGBoost"])
            
             # Option for hyperparameter tuning
            hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning")

            if hyperparameter_tuning:
                st.subheader("Hyperparameter Tuning")

                if model_choice == "Random Forest":
                    n_estimators = st.number_input("Number of Estimators", min_value=1, max_value=500, value=100, step=10)
                    max_depth = st.number_input("Max Depth", min_value=1, max_value=100, value=10, step=1)
                    min_samples_split = st.number_input("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
                    min_samples_leaf = st.number_input("Min Samples Leaf", min_value=1, max_value=20, value=1, step=1)
                    max_features = st.selectbox("Max Features", ["sqrt", "log2"])
                    if problem_type == "Classification":
                        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=42)
                    else:
                        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=42)

                elif model_choice in ["Logistic Regression", "Linear Regression"]:
                    if problem_type == "Classification":
                        penalty = st.selectbox("Penalty", ["l1", "l2", "elasticnet", "none"])
                        C = st.number_input("Inverse of regularization strength", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
                        solver = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"])
                        max_iter = st.number_input("Maximum Iterations", min_value=100, max_value=1000, value=100, step=50)
                        model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter, random_state=42)
                    else:
                        model = LinearRegression(fit_intercept=True, n_jobs=-1)
                        st.write("Hyperparameter is fit_intercept=True and n_jobs=-1")

                elif model_choice == "Support Vector Machine":
                    C = st.number_input("Regularization parameter", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
                    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                    gamma = st.selectbox("Kernel Coefficient", ["scale", "auto"])
                    degree = st.number_input("Degree (for poly kernel)", min_value=2, max_value=10, value=3, step=1)
                    if problem_type == "Classification":
                        model = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, random_state=42)
                    else:
                        model = SVR(C=C, kernel=kernel, gamma=gamma, degree=degree)

                elif model_choice == "Decision Tree":
                    max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=20, step=1)
                    min_samples_split = st.number_input("Min Samples Split", min_value=2, max_value=10, value=2, step=1)
                    min_samples_leaf = st.number_input("Min Samples Leaf", min_value=1, max_value=10, value=1, step=1)
                    criterion = st.selectbox("Criterion", ["gini", "entropy"])
                    if problem_type == "Classification":
                        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, 
                                                        min_samples_leaf=min_samples_leaf, criterion=criterion, random_state=42)

                elif model_choice == "K-Nearest Neighbors":
                    n_neighbors = st.number_input("Number of Neighbors", min_value=1, max_value=20, value=5, step=1)
                    weights = st.selectbox("Weights", ["uniform", "distance"])
                    algorithm = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
                    p = st.number_input("Power Parameter", min_value=1, max_value=2, value=2, step=1)
                    if problem_type == "Classification":
                        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p)
                    else:
                        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p)

                elif model_choice == "XGBoost":
                    n_estimators = st.number_input("Number of Estimators", min_value=1, max_value=500, value=100, step=10)
                    max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=6, step=1)
                    learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.3, step=0.01)
                    subsample = st.number_input("Subsample", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
                    colsample_bytree = st.number_input("Column Sample By Tree", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
                    if problem_type == "Classification":
                        model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                                  subsample=subsample, colsample_bytree=colsample_bytree, use_label_encoder=False, eval_metric='mlogloss')
                    else:
                        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                                 subsample=subsample, colsample_bytree=colsample_bytree)

            else:
                # Default hyperparameters
                if model_choice == "Random Forest":
                    if problem_type == "Classification":
                        model = RandomForestClassifier(random_state=42)
                    else:
                        model = RandomForestRegressor(random_state=42)

                elif model_choice in ["Logistic Regression", "Linear Regression"]:
                    if problem_type == "Classification":
                        model = LogisticRegression(random_state=42)
                    else:
                        model = LinearRegression()

                elif model_choice == "Support Vector Machine":
                    if problem_type == "Classification":
                        model = SVC(random_state=42)
                    else:
                        model = SVR()

                elif model_choice == "Decision Tree":
                    if problem_type == "Classification":
                        model = DecisionTreeClassifier(random_state=42)

                elif model_choice == "K-Nearest Neighbors":
                    if problem_type == "Classification":
                        model = KNeighborsClassifier()

                elif model_choice == "XGBoost":
                    if problem_type == "Classification":
                        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
                    else:
                        model = xgb.XGBRegressor()

            # Standardize features for models like SVC or Logistic Regression
            if model_choice in ["Logistic Regression", "Support Vector Machine", "Linear Regression"]:
                scaler = StandardScaler()
                X_resampled = scaler.fit_transform(X_resampled)        

            # Build model
            if st.button("Train Model"):
                start_time = time.time()
                 
                # Train model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                end_time = time.time()
                training_time = end_time - start_time
                st.info(f"Training time: {training_time:.2f} seconds")
                
                if problem_type == "Classification":
                    st.subheader("Classification Report")
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    df_report = pd.DataFrame(report).transpose()
                    st.write(df_report)
                    st.info(f'''
                            Model Accuracy: {accuracy:.2f} \n
                            Model Balanced Accuracy: {balanced_accuracy:.2f}
                            ''')

                    st.subheader("Confusion Matrix")
                    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)
                else:
                    st.subheader("Regression Metrics")
                    calculate_regression_metrics(y_test, y_pred)
                    plot_regression_results(y_test, y_pred)

                plot_and_interpret_learning_curve(model, X_resampled, y_resampled, cv=5)
                
                perform_hypothesis_test(y_test, y_pred)
                                
                st.write("---")
                # Save model
                st.subheader("Save Model")
                with open('model.pkl', 'wb') as file:
                    pickle.dump(model, file)
                st.success("Model trained and saved successfully!")

                with open('model.pkl', 'rb') as file:
                    model_bytes = file.read()
                st.download_button(
                    label="Download Model",
                    data=model_bytes,
                    file_name="model.pkl",
                    mime="application/octet-stream"
                )

if __name__ == "__main__":
    build_model()
