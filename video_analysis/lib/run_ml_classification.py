import pandas as pd
import numpy as np

from . import config
from . import ml_models as m2t

from matplotlib import pyplot as plt

from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize

import pickle
import os
import joblib


from sklearn.model_selection import LeaveOneOut, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    cohen_kappa_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

from datetime import datetime


def classify_video(dataset_identifier, features_type):
    """
    Perform video classification using multiple machine learning models with Leave-One-Out cross-validation, and generate evaluation metrics
    and visualizations.

    This function:
    - Reads diagnostic labels (UPDRS) for the specified dataset.
    - Reads extracted features corresponding to the dataset and feature type.
    - Filters the data to include only videos present in both diagnostics and features.
    - Trains and evaluates multiple models (from `m2t.models`) using Leave-One-Out CV.
    - Performs hyperparameter tuning with GridSearchCV for each model.
    - Calculates metrics including Accuracy, Acceptable Accuracy (prediction off by ≤1), Kappa, F1 score, MCC, and ROC AUC.
    - Saves individual predictions and results for each model.
    - Generates and saves aggregated confusion matrices for each model.
    - Saves a summary CSV of all evaluation metrics for all models.
    - Saves per-fold train/test accuracy for overfitting analysis.
    - Saves per-fold selected features for tsfresh/fi_tsfresh.

    Args:
        dataset_identifier (str): Name or identifier of the dataset to process.
        features_type (str): Type of features to use for classification ("classical", "tsfresh" or "fi_tsfresh").

    Returns:
        None: Results are saved to CSV files and confusion matrices are saved as PNGs specified by `config`.
    """

    # Ensure directories exist
    os.makedirs(config.models_files_dir, exist_ok=True)
    os.makedirs(config.checkpoint_files_dir, exist_ok=True)
    os.makedirs(config.results_files_dir + "pickle/", exist_ok=True)
    os.makedirs(config.results_files_dir + "confusion_matrix/", exist_ok=True)
    os.makedirs(config.results_files_dir + "roc_curves/", exist_ok=True)
    os.makedirs(config.results_files_dir + "feature_selection/", exist_ok=True)
    os.makedirs(config.results_files_dir + "overfit/", exist_ok=True)

    # Checkpoint path
    checkpoint_path = os.path.join(
        config.checkpoint_files_dir,
        f"{dataset_identifier}_{features_type}_checkpoint.pkl",
    )

    # Read diagnostic csv containing UPDRS ratings
    y_diagnostic = pd.read_csv(
        config.input_files_dir + dataset_identifier + "_diagnostic.csv",
        dtype={"UPDRS": np.int32, "ID": str},
    )
    y_diagnostic = y_diagnostic.sort_values(by=["ID"])
    y_diagnostic = y_diagnostic.drop_duplicates()
    y_diagnostic = y_diagnostic.reset_index()
    y_diagnostic = y_diagnostic.drop(columns=["index"])

    # Read features data and prepare it
    features = pd.read_csv(
        config.output_files_dir
        + dataset_identifier
        + "_"
        + features_type
        + "_features.csv",
        dtype={"Unnamed: 0": str},
    )
    features = features.rename(columns={"Unnamed: 0": "ID"})
    features = features.sort_values(by=["ID"])

    # Filter data to ensure that we have some ID in both sides due to probable video rejections
    list_id_y_diagnostic = np.unique(np.array(y_diagnostic["ID"]))
    list_id_data_final = np.unique(np.array(features["ID"]))
    features = features[features.ID.isin(list_id_y_diagnostic)]
    y_diagnostic = y_diagnostic[y_diagnostic.ID.isin(list_id_data_final)]

    # Save feature names for potential tsfresh selection
    feature_names = features.drop(columns=["ID"]).columns.tolist()

    # Build numpy array from features
    features_only = features.drop(columns=["ID"])
    final_x_array = np.array(features_only)

    # Build numpy array from diagnotics
    y_diagnostic_array1d = np.array(y_diagnostic["UPDRS"])

    # Load checkpoint if exists
    start_fold = 0
    feature_selection_rows = []
    train_test_rows = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)
            start_fold = checkpoint_data["last_fold"] + 1
            final_result = checkpoint_data["final_result"]
            conf_matrix_data = checkpoint_data["conf_matrix_data"]
            feature_selection_rows = checkpoint_data.get("feature_selection_rows", [])
            train_test_rows = checkpoint_data.get("train_test_rows", [])
            print(f"Resuming from fold {start_fold}")
    else:
        final_result = pd.DataFrame()
        # Variable for building confusion matrix
        conf_matrix_data = {
            m: {"y_true": [], "y_pred": [], "y_proba": []} for m in m2t.models.keys()
        }

    loo = LeaveOneOut()

    for fold_idx, (train_loo, test_loo) in enumerate(loo.split(final_x_array)):
        if fold_idx < start_fold:
            continue

        # Prepare data for this fold
        X_train = final_x_array[train_loo]
        X_test = final_x_array[test_loo]
        y_train = y_diagnostic_array1d[train_loo]
        y_test = y_diagnostic_array1d[test_loo]

        # Modification 1: Move select_features inside LOO loop for tsfresh/fi_tsfresh
        selected_cols = None
        if features_type in ["tsfresh", "fi_tsfresh"]:
            features_df_train = pd.DataFrame(X_train, columns=feature_names)
            y_train_series = pd.Series(y_train)

            # Select features based ONLY on training data
            selected_features_df = select_features(
                features_df_train,
                y_train_series,
                multiclass=True,
                n_significant=3,
                ml_task="classification",
            )
            selected_cols = selected_features_df.columns.tolist()

            for col in selected_cols:
                feature_selection_rows.append(
                    {
                        "feature_type": features_type,
                        "fold": fold_idx,
                        "feature": col,
                    }
                )

            # Update X_train and X_test with selected columns
            X_train = selected_features_df.values
            X_test = pd.DataFrame(X_test, columns=feature_names)[selected_cols].values

        # Apply Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for model_name, model in m2t.models.items():
            # Record execution starts
            now = datetime.now()
            execution_file = open(
                config.log_files_dir
                + dataset_identifier
                + "_"
                + features_type
                + "_execution.txt",
                "a",
            )
            execution_file.write(str(now) + "\t" + model_name + "\t")
            execution_file.close()

            model_param_grid = m2t.model_parameter_rules[model]

            gridS = GridSearchCV(
                estimator=model(),
                param_grid=model_param_grid,
                n_jobs=-1,
                cv=RepeatedStratifiedKFold(
                    n_splits=5, n_repeats=2, random_state=config.my_random_state
                ),
                scoring=config.grid_search_scoring_metric,
            )

            gridS.fit(X_train, y_train)

            # Save the best model
            model_save_path = os.path.join(
                config.models_files_dir,
                f"{dataset_identifier}_{features_type}_{model_name}_fold_{fold_idx}.joblib",
            )
            joblib.dump(gridS.best_estimator_, model_save_path)

            # Calculate A-AC
            y_pred = gridS.predict(X_test)
            train_pred = gridS.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, y_pred)
            substraction = abs(y_test - y_pred)
            total_hits = np.count_nonzero((substraction == 1) | (substraction == 0))
            length_array_prediction = substraction.size
            acceptable_accurary_pct = total_hits / length_array_prediction

            # We skip Kappa, F1 and MCC at fold level as they are invalid for a single sample
            data = {
                "Model": [model_name],
                "Variables": [str(gridS.best_estimator_).replace("\n", "")],
                "Accuracy": [test_acc],
                "Acceptable_Accuracy": [acceptable_accurary_pct],
                "value_real": [y_test[0]],
                "value_predict": [y_pred[0]],
            }

            final_result = pd.concat(
                [final_result, pd.DataFrame(data)], ignore_index=True
            )

            now2 = datetime.now()
            execution_file = open(
                config.log_files_dir
                + dataset_identifier
                + "_"
                + features_type
                + "_execution.txt",
                "a",
            )
            execution_file.write(str(now2 - now) + "\n")
            execution_file.close()

            # Save train/test accuracy per fold for overfitting analysis
            train_test_rows.append(
                {
                    "feature_type": features_type,
                    "fold": fold_idx,
                    "model": model_name,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "n_features": X_train.shape[1],
                }
            )

            # Save predictions and actual labels for this model
            conf_matrix_data[model_name]["y_true"].extend(y_test)
            conf_matrix_data[model_name]["y_pred"].extend(y_pred)

            # Save probabilities (for ROC AUC)
            if hasattr(gridS.best_estimator_, "predict_proba"):
                y_proba = gridS.predict_proba(X_test)
                conf_matrix_data[model_name]["y_proba"].extend(y_proba)

        # Save checkpoint after each fold
        with open(checkpoint_path, "wb") as f:
            pickle.dump(
                {
                    "last_fold": fold_idx,
                    "final_result": final_result,
                    "conf_matrix_data": conf_matrix_data,
                    "feature_selection_rows": feature_selection_rows,
                    "train_test_rows": train_test_rows,
                },
                f,
            )

    # Remove checkpoint after completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # Save Pickle with all predictions
    pickle_filename = f"{config.results_files_dir}pickle/{dataset_identifier}_{features_type}_raw_data.pkl"
    with open(pickle_filename, "wb") as f:
        pickle.dump(conf_matrix_data, f)

    # Save feature selection and train/test accuracy logs
    if feature_selection_rows:
        df_features_sel = pd.DataFrame(feature_selection_rows)
        features_sel_filename = (
            f"{config.results_files_dir}feature_selection/"
            f"{dataset_identifier}_{features_type}_selected_features_by_fold.csv"
        )
        df_features_sel.to_csv(features_sel_filename, index=False)

    if train_test_rows:
        df_train_test = pd.DataFrame(train_test_rows)
        train_test_filename = (
            f"{config.results_files_dir}overfit/"
            f"{dataset_identifier}_{features_type}_train_test_accuracy_by_fold.csv"
        )
        df_train_test.to_csv(train_test_filename, index=False)

    for model_name, data in conf_matrix_data.items():
        y_true_total = np.array(data["y_true"])
        y_pred_total = np.array(data["y_pred"])

        cm_total = confusion_matrix(y_true_total, y_pred_total)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm_total)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        for text in disp.text_.ravel():
            text.set_fontsize(16)

        ax.set_ylabel("Actual label", fontsize=16)
        ax.set_xlabel("Predicted label", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        # Save figure
        final_cm_filename = f"{config.results_files_dir}confusion_matrix/{dataset_identifier}_{features_type}_cm_{model_name}.png"
        plt.tight_layout()
        plt.savefig(final_cm_filename)
        plt.close()

    summary_results = []

    for model_name, data in conf_matrix_data.items():
        y_true_total = np.array(data["y_true"])
        y_pred_total = np.array(data["y_pred"])
        y_proba_total = np.array(data["y_proba"])

        # Classical metrix
        acc = accuracy_score(y_true_total, y_pred_total)
        kappa = cohen_kappa_score(y_true_total, y_pred_total)
        f1 = f1_score(y_true_total, y_pred_total, average="weighted")
        mcc = matthews_corrcoef(y_true_total, y_pred_total)

        precision = precision_score(y_true_total, y_pred_total, average="weighted")
        recall = recall_score(y_true_total, y_pred_total, average="weighted")

        # Percentage acceptable predict
        substraction = np.abs(y_true_total - y_pred_total)
        total_hits = np.count_nonzero((substraction == 1) | (substraction == 0))
        length_array_prediction = substraction.size
        acceptable_accurary_pct = (
            total_hits / length_array_prediction if length_array_prediction > 0 else 0
        )

        # ROC AUC (multiclass, with probabilities)
        try:
            roc_auc = roc_auc_score(
                y_true_total, y_proba_total, multi_class="ovr", average="weighted"
            )
        except Exception as e:
            roc_auc = None

        # Save summary results
        summary_results.append(
            {
                "Model": model_name,
                "Accuracy": acc,
                "Acceptable_accuracy": acceptable_accurary_pct,
                "Kappa_score": kappa,
                "F1_score": f1,
                "MCC": mcc,
                "ROC_AUC": roc_auc,
                "Precision": precision,
                "Recall": recall,
            }
        )

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_results)

    # Save summary CSCV
    summary_filename = f"{config.results_files_dir}{dataset_identifier}_{features_type}_execution_summary.csv"

    # Save CSV
    summary_df.to_csv(summary_filename, index=False, sep=";")

    final_result.to_csv(
        config.results_files_dir
        + dataset_identifier
        + "_"
        + features_type
        + "_result.csv",
        sep=";",
    )

    # Save ROC curves
    for model_name, data in conf_matrix_data.items():
        y_true_total = np.array(data["y_true"])
        y_proba_total = np.array(data["y_proba"])

        # Identify unique classes
        classes = np.unique(y_true_total)
        n_classes = len(classes)

        # Binarize labels for multi-class calculation (One-vs-Rest)
        y_true_binarized = label_binarize(y_true_total, classes=classes)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot a curve for each class
        for i in range(n_classes):
            RocCurveDisplay.from_predictions(
                y_true_binarized[:, i],
                y_proba_total[:, i],
                name=f"Class {classes[i]} vs Rest",
                ax=ax,
                plot_chance_level=(i == n_classes - 1),
            )

        # Customization
        ax.set_xlabel("False Positive Rate", fontsize=16)
        ax.set_ylabel("True Positive Rate", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)

        plt.legend(loc="lower right", fontsize=13)
        plt.tight_layout()

        # Save ROC curve figure
        final_roc_filename = f"{config.results_files_dir}roc_curves/{dataset_identifier}_{features_type}_roc_{model_name}.png"
        plt.savefig(final_roc_filename)
        plt.close()
