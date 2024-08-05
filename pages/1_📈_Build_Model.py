import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pickle
import time

st.set_page_config(
    page_title="Build Your Machine Learning Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

def plot_roc_auc(y_test, y_pred_proba, n_classes):
    plt.figure(figsize=(8, 5))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    else:
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f"ROC curve of class {i} (AUC = {roc_auc[i]:.2f})",
            )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    st.pyplot(plt)

    return roc_auc


def interpret_roc_auc(roc_auc):
    st.subheader("ROC AUC Interpretation")

    if isinstance(roc_auc, dict):
        avg_auc = sum(roc_auc.values()) / len(roc_auc)
        st.write(f"***Average AUC: {avg_auc:.2f}***")
    else:
        st.write(f"AUC: {roc_auc:.2f}")

    if isinstance(roc_auc, dict):
        auc_value = avg_auc
    else:
        auc_value = roc_auc

    if auc_value > 0.9:
        st.success("The model has excellent discriminative ability.")
    elif auc_value > 0.8:
        st.success("The model has very good discriminative ability.")
    elif auc_value > 0.7:
        st.info("The model has acceptable discriminative ability.")
    elif auc_value > 0.6:
        st.warning("The model has poor discriminative ability.")
    else:
        st.error("The model has failed to discriminate between classes.")

    st.write(
        "Note: An AUC of 0.5 suggests no discrimination (equivalent to random guessing)."
    )


def plot_distribution(data, x, y, title):
    distribution = pd.DataFrame({x: data.index, y: data.values})
    st.subheader(title)
    st.bar_chart(distribution, x=x, y=y, use_container_width=True)


def plot_learning_curve(model, X, y, cv):
    st.subheader("Plot and Interpretation of Learning Curve")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    plt.title("Learning Curve")
    plt.xlabel("Training Samples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    st.pyplot(plt)

    return train_scores_mean, test_scores_mean


def interpret_learning_curve(train_scores_mean, test_scores_mean):
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

    st.write("***Attention***")
    st.error(
        "This interpretation is subjective (author) and may not apply to all cases."
    )
    "---"


def plot_confusion_matrix(y_true, y_pred, classes):
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis_r",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(plt)


def plot_regression_results(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    st.pyplot(plt)


def calculate_regression_metrics(y_true, y_pred):
    st.subheader("Regression Metrics")
    r2 = r2_score(y_true, y_pred)
    n, p = len(y_true), len(y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    metrics = {
        "R-squared": r2,
        "Adjusted R-squared": adjusted_r2,
        "Mean Squared Error": mse,
        "Root Mean Squared Error": rmse,
        "Mean Absolute Error": mae,
    }

    for metric, value in metrics.items():
        st.write(f"{metric}: {value:.4f}")


def perform_hypothesis_test(y_true, y_pred):
    st.subheader("Hypothesis Testing")
    t_statistic, p_value = stats.ttest_rel(y_true, y_pred)

    st.write(
        "Null Hypothesis: No significant difference between true and predicted values."
    )
    st.write(
        "Alternative Hypothesis: Significant difference between true and predicted values."
    )
    st.write(f"T-statistic: {t_statistic:.4f}")
    st.write(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        st.success(
            "Reject the null hypothesis: The model has significant predictive power."
        )
    else:
        st.info(
            "Fail to reject the null hypothesis: The model does not have significant predictive power."
        )


def get_model(model_choice, problem_type, hyperparameters):
    models = {
        "Random Forest": (RandomForestClassifier, RandomForestRegressor),
        "Logistic Regression": (LogisticRegression, LinearRegression),
        "Support Vector Machine": (SVC, SVR),
        "Decision Tree": (DecisionTreeClassifier, None),
        "K-Nearest Neighbors": (KNeighborsClassifier, KNeighborsRegressor),
        "XGBoost": (xgb.XGBClassifier, xgb.XGBRegressor),
    }

    model_class = (
        models[model_choice][0]
        if problem_type == "Classification"
        else models[model_choice][1]
    )

    if model_class:
        return model_class(**hyperparameters)
    else:
        st.error(f"{model_choice} is not available for {problem_type}.")
        return None


def build_model():
    st.sidebar.title("Contact")
    st.sidebar.info(
        """
        **Contact Me:**
        - Email: mulinuhaa@gmail.com
    """
    )

    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file for training", type="csv")
    if uploaded_file is None:
        return

    data = pd.read_csv(uploaded_file)
    st.info("Data Preview")
    st.write(data.head())
    st.write("Rows: ", data.shape[0])

    st.write("---")
    st.subheader("Preprocess Data")
    features = st.multiselect("Select features", data.columns)
    target = st.selectbox("Select target variable", data.columns)

    if not features or not target:
        return

    if target in features:
        features.remove(target)

    selected_data = data[features + [target]]

    if st.checkbox("Drop NA values"):
        original_shape = selected_data.shape
        selected_data = selected_data.dropna()
        st.info(f"New Rows: {selected_data.shape[0]}")

        st.download_button(
            label="Download Clean Data",
            data=selected_data.to_csv(index=False),
            file_name="Clean data.csv",
            mime="csv",
        )

    X, y = selected_data[features], selected_data[target]

    st.info("Updated Data Preview")
    st.write(selected_data.head())

    st.write("---")

    if st.checkbox("Encode target labels"):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        st.subheader("Label Encoding")
        encoding_info = pd.DataFrame(
            {
                "Original": label_encoder.classes_,
                "Encoded": range(len(label_encoder.classes_)),
            }
        )
        st.write(encoding_info)

    plot_distribution(
        pd.Series(y).value_counts(), "Class", "Frequency", "Class Distribution"
    )

    resampling_choice = st.selectbox(
        "Choose resampling technique",
        ["None", "Oversampling", "Undersampling", "SMOTE"],
    )

    if resampling_choice != " ":
        if resampling_choice == "Oversampling":
            X_resampled, y_resampled = RandomOverSampler(random_state=42).fit_resample(
                X, y
            )
        elif resampling_choice == "Undersampling":
            X_resampled, y_resampled = RandomUnderSampler(random_state=42).fit_resample(
                X, y
            )
        elif resampling_choice == "SMOTE":
            X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y

        plot_distribution(
            pd.Series(y_resampled).value_counts(),
            "Class",
            "Frequency",
            f"{resampling_choice} Class Distribution",
        )

    st.write("---")

    st.subheader("Build Model")
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_size, random_state=42
    )

    problem_type = st.selectbox(
        "Choose problem type", [" ", "Classification", "Regression"]
    )

    if problem_type == " ":
        return

    model_choices = [
        "Random Forest",
        "Logistic Regression",
        "Support Vector Machine",
        "Decision Tree",
        "K-Nearest Neighbors",
        "XGBoost",
    ]
    model_choice = st.selectbox("Choose a model", [" "] + model_choices)

    if model_choice == " ":
        return

    hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning")

    hyperparameters = {}
    if hyperparameter_tuning:
        st.subheader("Hyperparameter Tuning")
        if model_choice == "Random Forest":
            n_estimators = st.number_input(
                "Number of Estimators", min_value=1, max_value=500, value=100, step=10
            )
            max_depth = st.number_input(
                "Max Depth", min_value=1, max_value=100, value=10, step=1
            )
            min_samples_split = st.number_input(
                "Min Samples Split", min_value=2, max_value=20, value=2, step=1
            )
            min_samples_leaf = st.number_input(
                "Min Samples Leaf", min_value=1, max_value=20, value=1, step=1
            )
            max_features = st.selectbox("Max Features", ["sqrt", "log2"])
            if problem_type == "Classification":
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42,
                )

        elif model_choice in ["Logistic Regression", "Linear Regression"]:
            if problem_type == "Classification":
                penalty = st.selectbox("Penalty", ["l1", "l2", "elasticnet", "none"])
                C = st.number_input(
                    "Inverse of regularization strength",
                    min_value=0.01,
                    max_value=10.0,
                    value=1.0,
                    step=0.01,
                )
                solver = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"])
                max_iter = st.number_input(
                    "Maximum Iterations",
                    min_value=100,
                    max_value=1000,
                    value=100,
                    step=50,
                )
                model = LogisticRegression(
                    penalty=penalty,
                    C=C,
                    solver=solver,
                    max_iter=max_iter,
                    random_state=42,
                )
            else:
                model = LinearRegression(fit_intercept=True, n_jobs=-1)
                st.write("Hyperparameter is fit_intercept=True and n_jobs=-1")

        elif model_choice == "Support Vector Machine":
            C = st.number_input(
                "Regularization parameter",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.01,
            )
            kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
            gamma = st.selectbox("Kernel Coefficient", ["scale", "auto"])
            degree = st.number_input(
                "Degree (for poly kernel)", min_value=2, max_value=10, value=3, step=1
            )
            if problem_type == "Classification":
                model = SVC(
                    C=C, kernel=kernel, gamma=gamma, degree=degree, random_state=42
                )
            else:
                model = SVR(C=C, kernel=kernel, gamma=gamma, degree=degree)

        elif model_choice == "Decision Tree":
            max_depth = st.number_input(
                "Max Depth", min_value=1, max_value=50, value=20, step=1
            )
            min_samples_split = st.number_input(
                "Min Samples Split", min_value=2, max_value=10, value=2, step=1
            )
            min_samples_leaf = st.number_input(
                "Min Samples Leaf", min_value=1, max_value=10, value=1, step=1
            )
            criterion = st.selectbox("Criterion", ["gini", "entropy"])
            if problem_type == "Classification":
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    random_state=42,
                )

        elif model_choice == "K-Nearest Neighbors":
            n_neighbors = st.number_input(
                "Number of Neighbors", min_value=1, max_value=20, value=5, step=1
            )
            weights = st.selectbox("Weights", ["uniform", "distance"])
            algorithm = st.selectbox(
                "Algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            )
            p = st.number_input(
                "Power Parameter", min_value=1, max_value=2, value=2, step=1
            )
            if problem_type == "Classification":
                model = KNeighborsClassifier(
                    n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p
                )
            else:
                model = KNeighborsRegressor(
                    n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p
                )

        elif model_choice == "XGBoost":
            n_estimators = st.number_input(
                "Number of Estimators", min_value=1, max_value=500, value=100, step=10
            )
            max_depth = st.number_input(
                "Max Depth", min_value=1, max_value=50, value=6, step=1
            )
            learning_rate = st.number_input(
                "Learning Rate", min_value=0.01, max_value=1.0, value=0.3, step=0.01
            )
            subsample = st.number_input(
                "Subsample", min_value=0.1, max_value=1.0, value=1.0, step=0.1
            )
            colsample_bytree = st.number_input(
                "Column Sample By Tree",
                min_value=0.1,
                max_value=1.0,
                value=1.0,
                step=0.1,
            )
            if problem_type == "Classification":
                model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                )

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
                    model = xgb.XGBClassifier(eval_metric="mlogloss")
                else:
                    model = xgb.XGBRegressor()

    model = get_model(model_choice, problem_type, hyperparameters)

    if model is None:
        return

    # Standardize features for specific models
    if model_choice in [
        "Logistic Regression",
        "Support Vector Machine",
        "Linear Regression",
    ]:
        scaler = StandardScaler()
        X_resampled = scaler.fit_transform(X_resampled)

    if st.button("Train Model"):
        start_time = time.time()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        training_time = time.time() - start_time
        st.info(f"Training time: {training_time:.2f} seconds")

        if problem_type == "Classification":
            st.subheader("Classification Report")

            accuracy = accuracy_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.write(df_report)
            st.subheader("Accuracy")
            st.info(
                f"""
                    Model Accuracy: {accuracy:.2f} \n 
                    Model Balanced Accuracy: {balanced_accuracy:.2f}"""
            )

            plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)

            st.subheader("ROC Curve and AUC")
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)
                n_classes = len(np.unique(y))
                roc_auc = plot_roc_auc(y_test, y_pred_proba, n_classes)
                interpret_roc_auc(roc_auc)
            else:
                st.warning(
                    "This model doesn't support probability predictions, so ROC curve cannot be plotted."
                )
        else:
            calculate_regression_metrics(y_test, y_pred)
            plot_regression_results(y_test, y_pred)

        train_scores_mean, test_scores_mean = plot_learning_curve(
            model, X_resampled, y_resampled, cv=5
        )
        interpret_learning_curve(train_scores_mean, test_scores_mean)

        perform_hypothesis_test(y_test, y_pred)

        st.write("---")
        st.subheader("Save Model")
        with open("model.pkl", "wb") as file:
            pickle.dump(model, file)
        st.success("Model trained and saved successfully!")

        with open("model.pkl", "rb") as file:
            model_bytes = file.read()
        st.download_button(
            label="Download Model",
            data=model_bytes,
            file_name="model.pkl",
            mime="application/octet-stream",
        )


if __name__ == "__main__":
    build_model()
