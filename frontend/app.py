import streamlit as st
import pandas as pd
import requests # To call backend API
import json # For parsing JSON and error details
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Backend URL ---
BACKEND_URL = "http://localhost:8000"

# --- Page Configuration ---
st.set_page_config(page_title="Algorithmic Bias Analysis Tool", layout="wide")
st.title("⚖️ Algorithmic Bias Detection and Mitigation Tool")
st.markdown("""
This tool allows you to upload a dataset, configure an analysis to detect bias,
and optionally apply a mitigation technique. The backend performs the computations.
""")

# --- Charting Function ---
def display_metrics_charts(results, analysis_config):
    if not results or "pre_mitigation" not in results:
        st.write("Insufficient data for generating charts.")
        return

    pre_metrics_data = results["pre_mitigation"]
    post_metrics_data = results.get("post_mitigation")

    # Determine mitigation method name for chart labels
    mitigation_method_name = "N/A"
    if post_metrics_data: # Only relevant if there's post_mitigation data
        if post_metrics_data.get("method"): # From PR, RW in current backend
            mitigation_method_name = post_metrics_data.get("method").replace("_", " ").title()
        elif post_metrics_data.get("method_details") and post_metrics_data["method_details"].get("name"): # From ROC
            mitigation_method_name = post_metrics_data["method_details"]["name"].replace("_", " ").title()
        elif analysis_config.get("mitigation_method"): # Fallback to config if method name not in results
            mitigation_method_name = analysis_config.get("mitigation_method").replace("_", " ").title()
        if mitigation_method_name == "None": # Should ideally not happen if post_metrics_data exists due to a method
            mitigation_method_name = "Mitigated"


    st.subheader("Visualizations")

    # --- Chart 1: Accuracy Comparison ---
    try:
        acc_labels = ['Pre-Mitigation']
        acc_values = [pre_metrics_data.get('accuracy', np.nan)] # Use np.nan for missing

        if post_metrics_data:
            acc_labels.append(f'Post-Mitigation ({mitigation_method_name})')
            acc_values.append(post_metrics_data.get('accuracy', np.nan))

        fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
        # Filter out NaN values for plotting to avoid errors with bar creation
        valid_indices_acc = [i for i, v in enumerate(acc_values) if not np.isnan(v)]
        if not valid_indices_acc: # If all accuracies are NaN
             ax_acc.text(0.5, 0.5, "Accuracy data N/A", ha='center', va='center')
        else:
            bars_acc = ax_acc.bar([acc_labels[i] for i in valid_indices_acc],
                                  [acc_values[i] for i in valid_indices_acc],
                                  color=['skyblue', 'lightcoral'][:len(valid_indices_acc)])
            ax_acc.set_ylabel('Accuracy Score')
            ax_acc.set_title('Model Accuracy Comparison')
            ax_acc.set_ylim(0, max(1.0, np.nanmax(acc_values) * 1.1 if not all(np.isnan(x) for x in acc_values) else 1.0) ) # Adjust ylim based on data or default to 1.0
            for bar in bars_acc:
                yval = bar.get_height()
                if not np.isnan(yval):
                    ax_acc.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')
        st.pyplot(fig_acc)
    except Exception as e:
        st.error(f"Error generating accuracy chart: {e}")


    # --- Chart 2: Fairness Metrics Comparison ---
    try:
        fm_pre = pre_metrics_data.get("fairness_metrics", {})
        if not fm_pre:
            st.write("No fairness metrics available to visualize.")
            return

        metric_names = list(fm_pre.keys())
        metric_names_display = [name.replace("_", " ").title() for name in metric_names]

        pre_values_raw = [fm_pre.get(m) for m in metric_names]
        # Convert non-numeric or None to NaN for plotting
        pre_values = [float(v) if isinstance(v, (int,float)) else np.nan for v in pre_values_raw]


        x = np.arange(len(metric_names_display))
        width = 0.35

        fig_fm, ax_fm = plt.subplots(figsize=(12, 7)) # Increased size

        rects1 = ax_fm.bar(x - width/2 if post_metrics_data else x, pre_values, width, label='Pre-Mitigation', color='deepskyblue')

        if post_metrics_data:
            fm_post = post_metrics_data.get("fairness_metrics", {})
            post_values_raw = [fm_post.get(m) for m in metric_names] # Ensure same order
            post_values = [float(v) if isinstance(v, (int,float)) else np.nan for v in post_values_raw]
            rects2 = ax_fm.bar(x + width/2, post_values, width, label=f'Post-Mitigation ({mitigation_method_name})', color='lightcoral')

        ax_fm.set_ylabel('Metric Value')
        ax_fm.set_title('Fairness Metrics Comparison', fontsize=16)
        ax_fm.set_xticks(x)
        ax_fm.set_xticklabels(metric_names_display, rotation=30, ha="right", fontsize=10)
        ax_fm.legend(fontsize=10)
        ax_fm.axhline(0, color='grey', lw=0.8)

        def autolabel(rects, ax):
            for rect in rects:
                height = rect.get_height()
                if np.isnan(height): continue
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3 if height >= 0 else -15),
                            textcoords="offset points",
                            ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

        autolabel(rects1, ax_fm)
        if post_metrics_data and 'rects2' in locals():
            autolabel(rects2, ax_fm)

        fig_fm.tight_layout()
        st.pyplot(fig_fm)
    except Exception as e:
        st.error(f"Error generating fairness metrics chart: {e}")


# --- Session State Initialization --- (moved after function def)
if 'dataframe' not in st.session_state: st.session_state['dataframe'] = None
if 'filename' not in st.session_state: st.session_state['filename'] = None
if 'analysis_config' not in st.session_state: st.session_state['analysis_config'] = None
if 'analysis_results' not in st.session_state: st.session_state['analysis_results'] = None
if 'uploaded_file_obj_id' not in st.session_state: st.session_state['uploaded_file_obj_id'] = None


# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Configuration")
st.sidebar.subheader("1. Upload Dataset")
uploaded_file_obj = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"], key="file_uploader_key")

if uploaded_file_obj is not None:
    new_file_id = id(uploaded_file_obj) # Heuristic to detect new file upload
    if st.session_state.get('uploaded_file_obj_id') != new_file_id:
        st.session_state['dataframe'] = None
        st.session_state['filename'] = None
        st.session_state['analysis_config'] = None
        st.session_state['analysis_results'] = None
        try:
            uploaded_file_obj.seek(0)
            df = pd.read_csv(uploaded_file_obj)
            st.session_state['dataframe'] = df
            st.session_state['filename'] = uploaded_file_obj.name
            st.session_state['uploaded_file_obj_id'] = new_file_id
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
            st.session_state['dataframe'] = None; st.session_state['filename'] = None
            st.session_state['uploaded_file_obj_id'] = None
elif st.session_state.get('uploaded_file_obj_id') is not None:
    st.session_state['dataframe'] = None; st.session_state['filename'] = None
    st.session_state['analysis_config'] = None; st.session_state['analysis_results'] = None
    st.session_state['uploaded_file_obj_id'] = None

if st.session_state.get('dataframe') is not None:
    with st.expander("View Uploaded Dataset Preview (First 100 rows)", expanded=False):
        st.dataframe(st.session_state['dataframe'].head(100))

st.sidebar.subheader("2. Configure Analysis")
if st.session_state.get('dataframe') is not None and st.session_state.get('filename') is not None:
    df_columns = st.session_state['dataframe'].columns.tolist()
    saved_config = st.session_state.get('analysis_config', {})

    target_column = st.sidebar.selectbox(
        "Target Column (Outcome)", options=df_columns,
        index=df_columns.index(saved_config.get('target_column', df_columns[0])) if saved_config.get('target_column') in df_columns else 0,
        help="Select the column that represents the outcome."
    )
    favorable_target_value = st.sidebar.text_input(
        "Favorable Value for Target", value=saved_config.get('favorable_target_value', ''),
        help="Value in target column considered 'favorable' (e.g., '1', 'Approved'). Case-sensitive."
    )
    st.sidebar.markdown("---")
    protected_attribute_column = st.sidebar.selectbox(
        "Protected Attribute Column", options=df_columns,
        index=df_columns.index(saved_config.get('protected_attribute_column', df_columns[1] if len(df_columns)>1 else df_columns[0])) if saved_config.get('protected_attribute_column') in df_columns else (1 if len(df_columns)>1 else 0),
        help="Select column with sensitive attributes (e.g., race, sex)."
    )
    privileged_value = st.sidebar.text_input(
        f"Privileged Value for '{protected_attribute_column}'", value=saved_config.get('privileged_value', ''),
        help=f"Value for the privileged group in '{protected_attribute_column}' (e.g., '1', 'Male'). Case-sensitive."
    )
    unprivileged_value = st.sidebar.text_input(
        f"Unprivileged Value for '{protected_attribute_column}'", value=saved_config.get('unprivileged_value', ''),
        help=f"Value for the unprivileged group in '{protected_attribute_column}' (e.g., '0', 'Female'). Case-sensitive."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Mitigation (Optional)")
    mitigation_method_options = [None, "reweighing", "prejudice_remover", "reject_option_classification"]
    mitigation_method = st.sidebar.selectbox(
        "Select Mitigation Method", options=mitigation_method_options,
        format_func=lambda x: "None" if x is None else x.replace("_", " ").title(),
        index=mitigation_method_options.index(saved_config.get('mitigation_method', None)) if saved_config.get('mitigation_method') in mitigation_method_options else 0,
        help="Choose a bias mitigation technique."
    )
    st.sidebar.markdown("---")

    if st.sidebar.button("🚀 Run Analysis", key="run_analysis_button"):
        if not all([target_column, favorable_target_value, protected_attribute_column, privileged_value, unprivileged_value]):
            st.sidebar.warning("Please fill in all required configuration fields above.")
        else:
            st.session_state['analysis_config'] = {
                "dataset_filename": st.session_state['filename'],
                "target_column": target_column,
                "favorable_target_value": favorable_target_value,
                "protected_attribute_column": protected_attribute_column,
                "privileged_value": privileged_value,
                "unprivileged_value": unprivileged_value,
                "mitigation_method": mitigation_method if mitigation_method != "None" else None
            }
            st.session_state['analysis_results'] = None

            current_df = st.session_state['dataframe']
            current_filename = st.session_state['filename']
            config = st.session_state['analysis_config']

            with st.spinner("🔄 Running analysis... This may take a moment..."):
                try:
                    csv_buffer = current_df.to_csv(index=False).encode()
                    files_for_upload = {'file': (current_filename, csv_buffer, 'text/csv')}
                    upload_response = requests.post(f"{BACKEND_URL}/upload_dataset/", files=files_for_upload)
                    upload_response.raise_for_status()
                    upload_result = upload_response.json()
                    backend_filename = upload_result.get("filename", current_filename)

                    analysis_payload = {
                        "dataset_filename": backend_filename,
                        "target_column": config["target_column"],
                        "protected_attribute_column": config["protected_attribute_column"],
                        "favorable_target_value": str(config["favorable_target_value"]),
                        "privileged_value": str(config["privileged_value"]),
                        "unprivileged_value": str(config["unprivileged_value"]),
                        "mitigation_method": config["mitigation_method"]
                    }
                    analyze_response = requests.post(f"{BACKEND_URL}/analyze_bias/", json=analysis_payload)
                    analyze_response.raise_for_status()
                    st.session_state['analysis_results'] = analyze_response.json()
                    st.success("✅ Analysis complete!")
                except requests.exceptions.ConnectionError:
                    st.error(f"🔌 Connection Error: Could not connect at {BACKEND_URL}.")
                except requests.exceptions.HTTPError as e:
                    error_detail = "No specific error message from backend."
                    try:
                        error_detail = e.response.json().get("detail", error_detail) if e.response else str(e)
                    except json.JSONDecodeError:
                        error_detail = e.response.text if e.response is not None and e.response.text else str(e)
                    st.error(f"API Error: {e.response.status_code if e.response is not None else 'N/A'} - {error_detail}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                if not st.session_state.get('analysis_results'): # Ensure it's None if any error
                    st.session_state['analysis_results'] = None
else:
    st.sidebar.info("ℹ️ Please upload a CSV dataset to configure and run the analysis.")

# --- Main Area for Results ---
st.header("📊 Analysis Results")

if st.session_state.get('analysis_results'):
    results_data = st.session_state['analysis_results']
    current_config = st.session_state.get('analysis_config', {})

    # Display textual metrics first
    st.subheader("Key Metrics")
    pre_mitigation_data = results_data.get("pre_mitigation", {})
    post_mitigation_data = results_data.get("post_mitigation")

    if pre_mitigation_data:
        st.markdown("#### Pre-Mitigation")
        st.metric(label="Accuracy", value=f"{pre_mitigation_data.get('accuracy', 0.0):.4f}")
        fm_pre = pre_mitigation_data.get("fairness_metrics", {})
        if fm_pre:
            cols_pre_metrics = st.columns(len(fm_pre))
            for i, (metric, value) in enumerate(fm_pre.items()):
                cols_pre_metrics[i].metric(label=metric.replace("_", " ").title(), value=f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
        else:
            st.text("No fairness metrics available.")
    else:
        st.warning("No pre-mitigation results available in the response.")

    if post_mitigation_data:
        st.markdown("---")
        st.markdown("#### Post-Mitigation")

        method_name = "N/A"
        if post_mitigation_data.get("method"): method_name = post_mitigation_data.get("method").replace("_", " ").title()
        elif post_mitigation_data.get("method_details"): method_name = post_mitigation_data["method_details"].get("name", "N/A").replace("_", " ").title()
        elif current_config.get("mitigation_method"): method_name = current_config.get("mitigation_method").replace("_", " ").title()
        st.markdown(f"**Method Used:** {method_name}")

        if post_mitigation_data.get("method_details"):
            details = post_mitigation_data.get("method_details")
            if "optimized_metric" in details: st.markdown(f"*Optimized Metric (ROC):* {details['optimized_metric']}")
            if "bounds" in details: st.markdown(f"*Metric Bounds (ROC):* `{details['bounds']}`")

        st.metric(label=f"Accuracy", value=f"{post_mitigation_data.get('accuracy', 0.0):.4f}")
        fm_post = post_mitigation_data.get("fairness_metrics", {})
        if fm_post:
            cols_post_metrics = st.columns(len(fm_post))
            for i, (metric, value) in enumerate(fm_post.items()):
                cols_post_metrics[i].metric(label=metric.replace("_", " ").title(), value=f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
        else:
            st.text("No fairness metrics available.")
    elif current_config and current_config.get("mitigation_method"):
        st.markdown("---")
        st.markdown("#### Post-Mitigation")
        st.warning(f"Mitigation method '{current_config.get('mitigation_method').replace('_', ' ').title()}' was selected, but no post-mitigation results were returned.")

    # Call the charting function here
    st.markdown("---") # Visual separator before charts
    display_metrics_charts(results_data, current_config)

    with st.expander("View Raw JSON Results", expanded=False):
        st.json(results_data)

elif st.session_state.get('analysis_config'):
    st.info("⚙️ Analysis is configured. Click 'Run Analysis' in the sidebar to process.")
    with st.expander("Current Configuration (Review)", expanded=False):
        st.json(st.session_state['analysis_config'])
elif st.session_state.get('filename'):
    st.info("ℹ️ Dataset uploaded. Configure parameters and click 'Run Analysis'.")
else:
    st.info("👋 Welcome! Upload a dataset to begin.")

st.markdown("---")
st.markdown("Bias Analysis Tool v0.3 | An AI Agent Collaboration")
