from typing import Optional, List, Any, Dict
import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.postprocessing import RejectOptionClassification


def convert_df_to_aif360_dataset(df: pd.DataFrame,
                                 label_name: str,
                                 protected_attribute_names: List[str],
                                 favorable_label: Any,
                                 unfavorable_label: Any,
                                 privileged_classes: List[List[Any]],
                                 unprivileged_classes: List[List[Any]],
                                 feature_names: Optional[List[str]] = None,
                                 labels_override: Optional[np.ndarray] = None,
                                 input_label_column_name: Optional[str] = None,
                                 score_names: Optional[List[str]] = None
                                 ) -> BinaryLabelDataset:
    """
    Converts a pandas DataFrame to an AIF360 BinaryLabelDataset.
    ... ( আগের ডকস্ট্রিং সংক্ষিপ্ততার জন্য কাটা হয়েছে, কিন্তু যুক্তি একই ) ...
    Args:
        ...
        score_names (Optional[List[str]]): List of column names for scores (probabilities).
    """
    df_internal = df.copy()
    source_label_col_for_mapping = input_label_column_name if input_label_column_name else label_name

    if labels_override is not None:
        df_internal[label_name] = labels_override
        if not (favorable_label == 1 and unfavorable_label == 0):
             df_internal[label_name] = df_internal[label_name].replace({favorable_label: 1, unfavorable_label: 0}).astype(int)
    elif source_label_col_for_mapping in df_internal.columns:
        if unfavorable_label is not None:
             df_internal[label_name] = df_internal[source_label_col_for_mapping].replace({favorable_label: 1, unfavorable_label: 0})
        else:
             df_internal[label_name] = df_internal[source_label_col_for_mapping].apply(lambda x: 1 if x == favorable_label else 0)
    else:
        raise ValueError(f"Label column '{source_label_col_for_mapping}' not found and labels_override not provided.")

    df_internal[label_name] = pd.to_numeric(df_internal[label_name])

    cols_to_keep = [label_name] + protected_attribute_names
    if feature_names:
        cols_to_keep.extend(feature_names)
    if score_names: # Add score columns to cols_to_keep if they exist in df_internal
        cols_to_keep.extend(s_name for s_name in score_names if s_name in df_internal.columns)

    # Ensure no duplicates and all are present in df_internal before selecting
    unique_cols_to_keep = []
    for col in cols_to_keep:
        if col not in unique_cols_to_keep and col in df_internal.columns:
            unique_cols_to_keep.append(col)

    df_for_bld = df_internal[unique_cols_to_keep]

    # Prepare scores for BinaryLabelDataset constructor
    # It expects scores as a 2D numpy array. If single score column, it should be shape (n_samples, 1)
    scores_array = None
    if score_names:
        valid_score_names = [s_name for s_name in score_names if s_name in df_for_bld.columns]
        if valid_score_names:
            scores_array = df_for_bld[valid_score_names].values
            if scores_array.ndim == 1:
                scores_array = scores_array.reshape(-1,1)

    bld = BinaryLabelDataset(
        df=df_for_bld,
        label_names=[label_name],
        protected_attribute_names=protected_attribute_names,
        favorable_label=1,
        unfavorable_label=0,
        privileged_protected_attributes=privileged_classes,
        unprivileged_protected_attributes=unprivileged_classes,
        feature_names=feature_names,
        scores=scores_array # Pass the scores array
    )
    return bld


def apply_reject_option_classification(dataset_true_test: BinaryLabelDataset,
                                       dataset_pred_test: BinaryLabelDataset,
                                       privileged_groups: List[Dict[str, Any]],
                                       unprivileged_groups: List[Dict[str, Any]],
                                       metric_name: str = "Statistical Parity Difference",
                                       metric_lb: float = -0.1,
                                       metric_ub: float = 0.1) -> BinaryLabelDataset:
    """
    Applies Reject Option Classification post-processing.
    Args:
        dataset_true_test: BinaryLabelDataset with true labels for the test set.
        dataset_pred_test: BinaryLabelDataset with predicted labels/scores from a base model.
                           This dataset *must* have scores populated.
        privileged_groups: Definitions for privileged groups.
        unprivileged_groups: Definitions for unprivileged groups.
        metric_name: Fairness metric to optimize for (e.g., "Statistical Parity Difference").
        metric_lb: Lower bound for the metric.
        metric_ub: Upper bound for the metric.
    Returns:
        BinaryLabelDataset: Dataset with predictions transformed by ROC.
    """
    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups,
                                     low_class_thresh=0.01, high_class_thresh=0.99,
                                     num_class_thresh=100, num_ROC_margin=50,
                                     metric_name=metric_name,
                                     metric_lb=metric_lb, metric_ub=metric_ub)

    ROC.fit(dataset_true_test, dataset_pred_test)
    dataset_transf_pred_test = ROC.predict(dataset_pred_test)
    return dataset_transf_pred_test


def train_prejudice_remover(dataset_orig_train: BinaryLabelDataset, # Signature kept same
                            sensitive_attr_name: str,
                            eta_value: float = 1.0) -> PrejudiceRemover:
    """
    Trains a PrejudiceRemover model.
    Args:
        dataset_orig_train (BinaryLabelDataset): The training dataset. Must contain features
                                                 and the sensitive attribute as one of the features.
                                                 Labels should be binarized (0/1).
        sensitive_attr_name (str): The name of the sensitive attribute column within the
                                   features of dataset_orig_train.
        eta_value (float): Regularization parameter for PrejudiceRemover.
    Returns:
        PrejudiceRemover: The trained model.
    """
    # PrejudiceRemover's class_attr is inferred from dataset_orig_train.label_names[0]
    pr_model = PrejudiceRemover(sensitive_attr=sensitive_attr_name, eta=eta_value)
    pr_model.fit(dataset_orig_train)
    return pr_model


def apply_reweighing(dataset_orig_train: BinaryLabelDataset, # Keep signature same as before for this one
                     privileged_groups: List[Dict[str, Any]],
                     unprivileged_groups: List[Dict[str, Any]]) -> BinaryLabelDataset:
    """
    Applies the Reweighing algorithm to the training dataset.

    Args:
        dataset_orig_train (BinaryLabelDataset): The original training dataset.
        privileged_groups (List[Dict[str, Any]]): Defines the privileged group.
                                                  Example: [{'sex': 1}]
        unprivileged_groups (List[Dict[str, Any]]): Defines the unprivileged group.
                                                    Example: [{'sex': 0}]

    Returns:
        BinaryLabelDataset: The transformed training dataset with instance weights.
    """
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(dataset_orig_train)
    return dataset_transf_train


def calculate_disparate_impact(dataset_true: BinaryLabelDataset,
                               dataset_pred: BinaryLabelDataset,
                               privileged_groups: List[Dict[str, Any]],
                               unprivileged_groups: List[Dict[str, Any]]) -> float:
    """
    Calculates Disparate Impact.
    DI = rate of favorable outcome for unprivileged group / rate of favorable outcome for privileged group.
    Values close to 1 are preferred. Value < 1 means privileged group has higher rate.
    A common threshold is > 0.8.
    """
    metric = ClassificationMetric(dataset_true, dataset_pred,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)
    # Handle potential division by zero or empty groups if AIF360 doesn't
    try:
        di_value = metric.disparate_impact()
    except ZeroDivisionError:
        # Or return a specific indicator like float('inf') or float('-inf') or None
        # depending on how you want to handle cases where a group has zero favorable outcomes.
        # AIF360 might return nan or inf in some cases.
        # For now, let AIF360's behavior dictate, but be aware.
        di_value = float('nan')
    return di_value


def calculate_statistical_parity_difference(dataset_true: BinaryLabelDataset,
                                            dataset_pred: BinaryLabelDataset,
                                            privileged_groups: List[Dict[str, Any]],
                                            unprivileged_groups: List[Dict[str, Any]]) -> float:
    """
    Calculates Statistical Parity Difference (SPD).
    SPD = rate of favorable outcome for unprivileged group - rate of favorable outcome for privileged group.
    Values close to 0 are preferred. Positive value means unprivileged group has higher rate.
    """
    metric = ClassificationMetric(dataset_true, dataset_pred,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)
    # AIF360's mean_difference is equivalent to statistical_parity_difference
    return metric.mean_difference()


def calculate_equal_opportunity_difference(dataset_true: BinaryLabelDataset,
                                           dataset_pred: BinaryLabelDataset,
                                           privileged_groups: List[Dict[str, Any]],
                                           unprivileged_groups: List[Dict[str, Any]]) -> float:
    """
    Calculates Equal Opportunity Difference (EOD).
    EOD = True Positive Rate (TPR) for unprivileged group - TPR for privileged group.
    Measures equality of opportunity for positive outcomes among those who should receive them.
    Values close to 0 are preferred.
    """
    metric = ClassificationMetric(dataset_true, dataset_pred,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)
    return metric.equal_opportunity_difference()


def calculate_average_odds_difference(dataset_true: BinaryLabelDataset,
                                      dataset_pred: BinaryLabelDataset,
                                      privileged_groups: List[Dict[str, Any]],
                                      unprivileged_groups: List[Dict[str, Any]]) -> float:
    """
    Calculates Average Odds Difference.
    AOD = 0.5 * [(FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv)]
    Measures equality of opportunity considering both false positives and true positives.
    Values close to 0 are preferred.
    """
    metric = ClassificationMetric(dataset_true, dataset_pred,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)
    return metric.average_odds_difference()

# Placeholder for model prediction logic (to be developed later)
# For now, dataset_pred (BinaryLabelDataset with predicted labels)
# will be generated externally and passed to these functions.

# Example usage (will require actual data and model predictions):
# if __name__ == '__main__':
#     # This is a conceptual example.
#     # 1. Load your data into a pandas DataFrame
#     # example_data = {'feature1': [10, 20, 30, 40, 50, 60],
#     #                 'race': [0, 1, 0, 1, 0, 1], # 0 for unprivileged, 1 for privileged
#     #                 'true_label': [0, 1, 1, 0, 1, 0], # True outcomes
#     #                 'predicted_label': [0, 0, 1, 1, 1, 0]} # Model's predictions
#     # df = pd.DataFrame(example_data)

#     # favorable_label_value = 1
#     # unfavorable_label_value = 0
#     # label_column_name = 'true_label'
#     # protected_attribute_column_name = 'race'

#     # # Define privileged and unprivileged groups based on 'race' column
#     # # For AIF360, these are lists of dictionaries.
#     # # Example: race = 1 is privileged, race = 0 is unprivileged
#     # privileged_groups_def = [{protected_attribute_column_name: 1}]
#     # unprivileged_groups_def = [{protected_attribute_column_name: 0}]

    # # Convert DataFrame to AIF360 BinaryLabelDataset for true labels
#     # dataset_true_aif360 = convert_df_to_aif360_dataset(
#     #     df,
#     #     label_name=label_column_name,
#     #     protected_attribute_names=[protected_attribute_column_name],
#     #     favorable_label=favorable_label_value,
#     #     unfavorable_label=unfavorable_label_value,
#     #     # For BinaryLabelDataset, privileged_classes and unprivileged_classes
#     #     # expect a list of lists of attribute values.
#     #     # e.g. if 'race' is the protected attribute, and 1 is privileged value for 'race'
#     #     privileged_classes_def_aif = [[1]],
#     #     unprivileged_classes_def_aif = [[0]]
#     # )

#     # # For predicted labels, we need another BinaryLabelDataset.
#     # # This usually involves taking the 'predicted_label' column, and the original protected attributes.
#     # # Create a new DataFrame for predictions or modify the existing one for conversion.
#     # df_pred = df.copy()
#     # df_pred['label_for_pred_dataset'] = df_pred['predicted_label'] # Use predicted labels

#     # dataset_pred_aif360 = convert_df_to_aif360_dataset(
#     #     df_pred,
#     #     label_name='label_for_pred_dataset', # This now points to predicted_label values
#     #     protected_attribute_names=[protected_attribute_column_name],
#     #     favorable_label=favorable_label_value,
#     #     unfavorable_label=unfavorable_label_value,
#     #     privileged_classes_def_aif = [[1]],
#     #     unprivileged_classes_def_aif = [[0]]
#     # )

#     # # Now calculate metrics
#     # di = calculate_disparate_impact(dataset_true_aif360, dataset_pred_aif360, privileged_groups_def, unprivileged_groups_def)
#     # spd = calculate_statistical_parity_difference(dataset_true_aif360, dataset_pred_aif360, privileged_groups_def, unprivileged_groups_def)
#     # eod = calculate_equal_opportunity_difference(dataset_true_aif360, dataset_pred_aif360, privileged_groups_def, unprivileged_groups_def)
#     # aod = calculate_average_odds_difference(dataset_true_aif360, dataset_pred_aif360, privileged_groups_def, unprivileged_groups_def)

#     # print(f"Disparate Impact: {di}")
#     # print(f"Statistical Parity Difference: {spd}")
#     # print(f"Equal Opportunity Difference: {eod}")
#     # print(f"Average Odds Difference: {aod}")
pass
