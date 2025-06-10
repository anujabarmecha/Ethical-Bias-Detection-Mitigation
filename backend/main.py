import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from backend.bias_analysis import (
    convert_df_to_aif360_dataset,
    calculate_disparate_impact,
    calculate_statistical_parity_difference,
    calculate_equal_opportunity_difference,
    calculate_average_odds_difference,
    apply_reweighing,
    train_prejudice_remover,
    apply_reject_option_classification # New import
)

app = FastAPI()

DATASETS_DIR = "datasets"
os.makedirs(DATASETS_DIR, exist_ok=True)

class BiasAnalysisRequest(BaseModel):
    dataset_filename: str
    target_column: str
    protected_attribute_column: str
    favorable_target_value: Any
    privileged_value: Any
    unprivileged_value: Any
    mitigation_method: Optional[str] = None
    # Optional parameters for ROC, could be added later:
    # roc_metric_name: Optional[str] = "Statistical Parity Difference"
    # roc_metric_lb: Optional[float] = -0.05
    # roc_metric_ub: Optional[float] = 0.05


@app.post("/upload_dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are allowed.")
    file_path = os.path.join(DATASETS_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        if hasattr(file, 'file') and file.file:
            file.file.close()
    return {"message": "Dataset uploaded successfully", "filename": file.filename}


def try_convert_to_typed_value(value_str: str, series_for_type_check: pd.Series):
    if pd.api.types.is_object_dtype(series_for_type_check) or \
       pd.api.types.is_string_dtype(series_for_type_check) or \
       not pd.api.types.is_numeric_dtype(series_for_type_check):
        return str(value_str)
    try:
        if series_for_type_check.dtype.kind in 'iu':
             return int(value_str)
        elif series_for_type_check.dtype.kind in 'f':
             return float(value_str)
        else: # General numeric
            try:
                return int(value_str)
            except ValueError:
                return float(value_str)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Value '{value_str}' cannot be converted to the numeric type of column '{series_for_type_check.name}'.")


@app.post("/analyze_bias/")
async def analyze_bias(request: BiasAnalysisRequest):
    dataset_path = os.path.join(DATASETS_DIR, request.dataset_filename)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {request.dataset_filename}")

    try:
        df_orig = pd.read_csv(dataset_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {e}")

    if request.target_column not in df_orig.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found.")
    if request.protected_attribute_column not in df_orig.columns:
        raise HTTPException(status_code=400, detail=f"Protected attribute column '{request.protected_attribute_column}' not found.")

    try:
        favorable_target_value_parsed = try_convert_to_typed_value(str(request.favorable_target_value), df_orig[request.target_column])
        privileged_value_parsed = try_convert_to_typed_value(str(request.privileged_value), df_orig[request.protected_attribute_column])
        unprivileged_value_parsed = try_convert_to_typed_value(str(request.unprivileged_value), df_orig[request.protected_attribute_column])
    except HTTPException as e:
        raise e

    df_orig['target_binary'] = (df_orig[request.target_column] == favorable_target_value_parsed).astype(int)

    feature_columns = [col for col in df_orig.columns if col not in [request.target_column, request.protected_attribute_column, 'target_binary']]
    if not feature_columns:
        raise HTTPException(status_code=400, detail="No feature columns found.")

    df_train, df_test = train_test_split(df_orig, test_size=0.3, random_state=42, stratify=df_orig['target_binary'])
    df_train = df_train.copy()
    df_test = df_test.copy()

    aif_privileged_groups_orig = [{request.protected_attribute_column: privileged_value_parsed}]
    aif_unprivileged_groups_orig = [{request.protected_attribute_column: unprivileged_value_parsed}]
    aif_privileged_classes_orig = [[privileged_value_parsed]]
    aif_unprivileged_classes_orig = [[unprivileged_value_parsed]]

    X_train_pre_df = df_train[feature_columns]
    y_train_pre = df_train['target_binary']
    X_test_pre_df = df_test[feature_columns]
    y_test_pre = df_test['target_binary']

    categorical_features = X_train_pre_df.select_dtypes(include=['object', 'category']).columns.tolist()
    X_train_pre_processed = pd.get_dummies(X_train_pre_df, columns=categorical_features, drop_first=True)
    X_test_pre_processed = pd.get_dummies(X_test_pre_df, columns=categorical_features, drop_first=True)

    train_processed_cols = X_train_pre_processed.columns.tolist()
    X_test_pre_processed = X_test_pre_processed.reindex(columns=train_processed_cols, fill_value=0)

    scaler_pre = StandardScaler()
    X_train_pre_scaled = scaler_pre.fit_transform(X_train_pre_processed)
    X_test_pre_scaled = scaler_pre.transform(X_test_pre_processed)

    model_pre = LogisticRegression(solver='liblinear', random_state=42)
    model_pre.fit(X_train_pre_scaled, y_train_pre)

    y_pred_labels_pre_test = model_pre.predict(X_test_pre_scaled)
    y_pred_proba_pre_test = model_pre.predict_proba(X_test_pre_scaled)[:, 1] # Probabilities for favorable class
    accuracy_pre = accuracy_score(y_test_pre, y_pred_labels_pre_test)

    df_test_for_aif_pre = df_test.copy()
    df_test_for_aif_pre['true_labels_aif'] = y_test_pre
    df_test_for_aif_pre['predicted_labels_aif_pre'] = y_pred_labels_pre_test
    df_test_for_aif_pre['scores_aif_pre'] = y_pred_proba_pre_test # Add scores column

    try:
        dataset_true_test_pre = convert_df_to_aif360_dataset(
            df=df_test_for_aif_pre, input_label_column_name='true_labels_aif', label_name='label',
            protected_attribute_names=[request.protected_attribute_column],
            favorable_label=1, unfavorable_label=0,
            privileged_classes=aif_privileged_classes_orig, unprivileged_classes=aif_unprivileged_classes_orig,
            feature_names=feature_columns
        )
        dataset_pred_test_pre = convert_df_to_aif360_dataset(
            df=df_test_for_aif_pre, input_label_column_name='predicted_labels_aif_pre', label_name='label',
            protected_attribute_names=[request.protected_attribute_column],
            favorable_label=1, unfavorable_label=0,
            privileged_classes=aif_privileged_classes_orig, unprivileged_classes=aif_unprivileged_classes_orig,
            feature_names=feature_columns,
            score_names=['scores_aif_pre'] # Pass score column name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting pre-mitigation data to AIF360: {str(e)}")

    pre_mitigation_fairness_metrics = {
        "disparate_impact": calculate_disparate_impact(dataset_true_test_pre, dataset_pred_test_pre, aif_privileged_groups_orig, aif_unprivileged_groups_orig),
        "statistical_parity_difference": calculate_statistical_parity_difference(dataset_true_test_pre, dataset_pred_test_pre, aif_privileged_groups_orig, aif_unprivileged_groups_orig),
        "equal_opportunity_difference": calculate_equal_opportunity_difference(dataset_true_test_pre, dataset_pred_test_pre, aif_privileged_groups_orig, aif_unprivileged_groups_orig),
        "average_odds_difference": calculate_average_odds_difference(dataset_true_test_pre, dataset_pred_test_pre, aif_privileged_groups_orig, aif_unprivileged_groups_orig)
    }
    results = {
        "pre_mitigation": {
            "accuracy": accuracy_pre,
            "fairness_metrics": pre_mitigation_fairness_metrics,
            "aif_datasets": { # Store datasets for ROC
                "true_test": dataset_true_test_pre,
                "pred_test": dataset_pred_test_pre
            }
        },
        "post_mitigation": None
    }

    if request.mitigation_method == "reweighing":
        # ... (Reweighing code remains unchanged from previous step)
        try:
            dataset_orig_train_rw = convert_df_to_aif360_dataset(
                df=df_train.copy(), input_label_column_name='target_binary', label_name='target_binary',
                protected_attribute_names=[request.protected_attribute_column],
                favorable_label=1, unfavorable_label=0,
                privileged_classes=aif_privileged_classes_orig, unprivileged_classes=aif_unprivileged_classes_orig,
                feature_names=feature_columns
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error converting training data for reweighing: {str(e)}")

        dataset_transf_train_rw = apply_reweighing(dataset_orig_train_rw, aif_privileged_groups_orig, aif_unprivileged_groups_orig)
        X_train_post_rw_df = pd.DataFrame(dataset_transf_train_rw.features, columns=dataset_transf_train_rw.feature_names)
        y_train_post_rw = dataset_transf_train_rw.labels.ravel()
        sample_weights_post_rw = dataset_transf_train_rw.instance_weights.ravel()

        X_train_post_rw_processed = pd.get_dummies(X_train_post_rw_df, columns=categorical_features, drop_first=True)
        X_train_post_rw_processed = X_train_post_rw_processed.reindex(columns=train_processed_cols, fill_value=0)
        X_train_post_rw_scaled = scaler_pre.transform(X_train_post_rw_processed)

        model_post_rw = LogisticRegression(solver='liblinear', random_state=42)
        model_post_rw.fit(X_train_post_rw_scaled, y_train_post_rw, sample_weight=sample_weights_post_rw)
        y_pred_post_rw_test = model_post_rw.predict(X_test_pre_scaled)
        accuracy_post_rw = accuracy_score(y_test_pre, y_pred_post_rw_test)

        df_test_for_aif_post_rw = df_test.copy()
        # Add predicted labels and scores for post-reweighing model to enable ROC on its output if desired (though not primary path here)
        df_test_for_aif_post_rw['predicted_labels_aif_post_rw'] = y_pred_post_rw_test
        y_pred_proba_post_rw_test = model_post_rw.predict_proba(X_test_pre_scaled)[:,1]
        df_test_for_aif_post_rw['scores_aif_post_rw'] = y_pred_proba_post_rw_test

        try:
            # Create a new dataset_true_test_post_rw if needed, or reuse dataset_true_test_pre
            # dataset_true_test_post_rw = dataset_true_test_pre (if protected attributes and features are identical)
            dataset_pred_test_post_rw = convert_df_to_aif360_dataset(
                df=df_test_for_aif_post_rw, input_label_column_name='predicted_labels_aif_post_rw', label_name='label',
                protected_attribute_names=[request.protected_attribute_column],
                favorable_label=1, unfavorable_label=0,
                privileged_classes=aif_privileged_classes_orig, unprivileged_classes=aif_unprivileged_classes_orig,
                feature_names=feature_columns,
                score_names=['scores_aif_post_rw'] # Include scores for this dataset too
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error converting post-reweighing data to AIF360: {str(e)}")

        post_rw_fairness = {
            "disparate_impact": calculate_disparate_impact(dataset_true_test_pre, dataset_pred_test_post_rw, aif_privileged_groups_orig, aif_unprivileged_groups_orig),
            "statistical_parity_difference": calculate_statistical_parity_difference(dataset_true_test_pre, dataset_pred_test_post_rw, aif_privileged_groups_orig, aif_unprivileged_groups_orig),
            "equal_opportunity_difference": calculate_equal_opportunity_difference(dataset_true_test_pre, dataset_pred_test_post_rw, aif_privileged_groups_orig, aif_unprivileged_groups_orig),
            "average_odds_difference": calculate_average_odds_difference(dataset_true_test_pre, dataset_pred_test_post_rw, aif_privileged_groups_orig, aif_unprivileged_groups_orig)
        }
        results["post_mitigation"] = {"accuracy": accuracy_post_rw, "fairness_metrics": post_rw_fairness, "method": "Reweighing"}


    elif request.mitigation_method == "prejudice_remover":
        # ... (Prejudice Remover code remains unchanged from previous step)
        X_train_pr_features_df = pd.DataFrame(X_train_pre_scaled, columns=train_processed_cols, index=df_train.index)
        binarized_pa_train = df_train[request.protected_attribute_column].apply(lambda x: 1 if x == privileged_value_parsed else 0).values
        protected_attr_col_for_pr = f"{request.protected_attribute_column}_binarized_for_pr"
        X_train_pr_features_df[protected_attr_col_for_pr] = binarized_pa_train
        pr_feature_names = train_processed_cols + [protected_attr_col_for_pr]

        try:
            dataset_train_for_pr = convert_df_to_aif360_dataset(
                df=X_train_pr_features_df, label_name="target_for_pr", labels_override=y_train_pre.values,
                protected_attribute_names=[protected_attr_col_for_pr], favorable_label=1, unfavorable_label=0,
                privileged_classes=[[1]], unprivileged_classes=[[0]], feature_names=pr_feature_names
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error converting training data for Prejudice Remover: {str(e)}")
        eta_value = 1.0
        pr_model = train_prejudice_remover(dataset_train_for_pr, sensitive_attr_name=protected_attr_col_for_pr, eta_value=eta_value)

        X_test_pr_features_df = pd.DataFrame(X_test_pre_scaled, columns=train_processed_cols, index=df_test.index)
        binarized_pa_test = df_test[request.protected_attribute_column].apply(lambda x: 1 if x == privileged_value_parsed else 0).values
        X_test_pr_features_df[protected_attr_col_for_pr] = binarized_pa_test

        try:
            dataset_test_for_pr_pred = convert_df_to_aif360_dataset(
                df=X_test_pr_features_df, label_name="target_for_pr", labels_override=y_test_pre.values,
                protected_attribute_names=[protected_attr_col_for_pr], favorable_label=1, unfavorable_label=0,
                privileged_classes=[[1]], unprivileged_classes=[[0]], feature_names=pr_feature_names
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error converting test data for PR prediction: {str(e)}")

        dataset_pred_post_pr = pr_model.predict(dataset_test_for_pr_pred)
        y_pred_post_pr_test = dataset_pred_post_pr.labels.ravel()
        accuracy_post_pr = accuracy_score(y_test_pre, y_pred_post_pr_test)

        df_test_for_aif_post_pr = X_test_pr_features_df.copy()
        df_test_for_aif_post_pr['true_labels'] = y_test_pre.values
        df_test_for_aif_post_pr['predicted_labels_pr'] = y_pred_post_pr_test
        # For PR, scores might not be readily available from its .predict(), or might need specific handling if they are.
        # If PR model outputs scores in dataset_pred_post_pr.scores, they could be used here.
        # For now, assuming label-based metrics.

        try:
            dataset_true_test_post_pr = convert_df_to_aif360_dataset(
                df=df_test_for_aif_post_pr, input_label_column_name='true_labels', label_name='label',
                protected_attribute_names=[protected_attr_col_for_pr], favorable_label=1, unfavorable_label=0,
                privileged_classes=[[1]], unprivileged_classes=[[0]], feature_names=pr_feature_names
            )
            dataset_pred_test_post_pr_metrics = convert_df_to_aif360_dataset(
                df=df_test_for_aif_post_pr, input_label_column_name='predicted_labels_pr', label_name='label',
                protected_attribute_names=[protected_attr_col_for_pr], favorable_label=1, unfavorable_label=0,
                privileged_classes=[[1]], unprivileged_classes=[[0]], feature_names=pr_feature_names
                # score_names could be added if dataset_pred_post_pr has reliable scores
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error converting data for post-PR metrics: {str(e)}")

        pr_privileged_groups = [{protected_attr_col_for_pr: 1}]
        pr_unprivileged_groups = [{protected_attr_col_for_pr: 0}]
        post_pr_fairness = {
            "disparate_impact": calculate_disparate_impact(dataset_true_test_post_pr, dataset_pred_test_post_pr_metrics, pr_privileged_groups, pr_unprivileged_groups),
            "statistical_parity_difference": calculate_statistical_parity_difference(dataset_true_test_post_pr, dataset_pred_test_post_pr_metrics, pr_privileged_groups, pr_unprivileged_groups),
            "equal_opportunity_difference": calculate_equal_opportunity_difference(dataset_true_test_post_pr, dataset_pred_test_post_pr_metrics, pr_privileged_groups, pr_unprivileged_groups),
            "average_odds_difference": calculate_average_odds_difference(dataset_true_test_post_pr, dataset_pred_test_post_pr_metrics, pr_privileged_groups, pr_unprivileged_groups)
        }
        results["post_mitigation"] = {"accuracy": accuracy_post_pr, "fairness_metrics": post_pr_fairness, "method": "PrejudiceRemover"}

    elif request.mitigation_method == "reject_option_classification":
        if 'pre_mitigation' not in results or 'aif_datasets' not in results['pre_mitigation']:
            raise HTTPException(status_code=500, detail="Pre-mitigation AIF360 datasets not found for Reject Option Classification.")

        dataset_true_for_roc = results['pre_mitigation']['aif_datasets']['true_test']
        dataset_pred_for_roc = results['pre_mitigation']['aif_datasets']['pred_test'] # This now has scores

        # Default ROC parameters (can be made configurable via request)
        roc_metric_name = "Statistical Parity Difference"
        roc_metric_lb = -0.05
        roc_metric_ub = 0.05

        try:
            dataset_transf_pred_post_roc = apply_reject_option_classification(
                dataset_true_test=dataset_true_for_roc,
                dataset_pred_test=dataset_pred_for_roc,
                privileged_groups=aif_privileged_groups_orig,
                unprivileged_groups=aif_unprivileged_groups_orig,
                metric_name=roc_metric_name,
                metric_lb=roc_metric_lb,
                metric_ub=roc_metric_ub
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error applying Reject Option Classification: {str(e)}. Ensure pre-mitigation model produced scores.")

        y_pred_post_roc_test = dataset_transf_pred_post_roc.labels.ravel()
        accuracy_post_roc = accuracy_score(dataset_true_for_roc.labels.ravel(), y_pred_post_roc_test)

        # Fairness metrics use dataset_true_for_roc and the transformed predictions
        post_roc_fairness = {
            "disparate_impact": calculate_disparate_impact(dataset_true_for_roc, dataset_transf_pred_post_roc, aif_privileged_groups_orig, aif_unprivileged_groups_orig),
            "statistical_parity_difference": calculate_statistical_parity_difference(dataset_true_for_roc, dataset_transf_pred_post_roc, aif_privileged_groups_orig, aif_unprivileged_groups_orig),
            "equal_opportunity_difference": calculate_equal_opportunity_difference(dataset_true_for_roc, dataset_transf_pred_post_roc, aif_privileged_groups_orig, aif_unprivileged_groups_orig),
            "average_odds_difference": calculate_average_odds_difference(dataset_true_for_roc, dataset_transf_pred_post_roc, aif_privileged_groups_orig, aif_unprivileged_groups_orig)
        }
        results["post_mitigation"] = {
            "accuracy": accuracy_post_roc,
            "fairness_metrics": post_roc_fairness,
            "method": "RejectOptionClassification",
            "method_details": {
                "optimized_metric": roc_metric_name,
                "metric_bounds": {"lower": roc_metric_lb, "upper": roc_metric_ub}
            }
        }

    return results

# uvicorn backend.main:app --reload
