import numpy as np
import pickle
import os
import pandas as pd


def find_max_likelihood_full_model(path, smi_value=None, beta_value=None):
    max_log_likelihood = -np.Inf
    for batch in os.listdir(path):
        df = pd.read_csv(os.path.join(path, batch))
        # find the parameter set with the maximal likelihood
        if smi_value is None and beta_value is None:
            df.sort_values("log-likelihood", ascending=False)
            if df["log-likelihood"][0] > max_log_likelihood:
                max_log_likelihood = df["log-likelihood"][0]
                max_smi = df["smi"][0]
                max_beta = df["beta"][0]
            if np.isnan(df["log-likelihood"][0]) or -np.Inf == df["log-likelihood"][0]:
                print(batch)
        # find the parameter set with the maximal likelihood under the constraint that smi is smi_value
        elif smi_value is not None and beta_value is None:
            if smi_value not in df["smi"].values:
                continue
            for run in range(df.shape[0]):
                if df["log-likelihood"][run] > max_log_likelihood and smi_value == df["smi"][run]:
                    max_log_likelihood = df["log-likelihood"][run]
                    max_smi = df["smi"][run]
                    max_beta = df["beta"][run]
        elif smi_value is None and beta_value is not None:
            if beta_value not in df["beta"].values:
                continue
            for run in range(df.shape[0]):
                if df["log-likelihood"][run] > max_log_likelihood and beta_value == df["beta"][run]:
                    max_log_likelihood = df["log-likelihood"][run]
                    max_smi = df["smi"][run]
                    max_beta = df["beta"][run]
        else:
            if smi_value not in df["smi"].values or beta_value not in df["beta"].values:
                continue
            for run in range(df.shape[0]):
                if df["log-likelihood"][run] > max_log_likelihood and smi_value == df["smi"][run] and beta_value == \
                        df["beta"][run]:
                    max_log_likelihood = df["log-likelihood"][run]
                    max_smi = df["smi"][run]
                    max_beta = df["beta"][run]
    return max_smi, max_beta, max_log_likelihood


def find_max_likelihood_distance_model(path):
    max_log_likelihood = -np.Inf
    for batch in os.listdir(path):
        df = pd.read_csv(os.path.join(path, batch))
        # find the parameter set with the maximal likelihood
        df.sort_values("log-likelihood", ascending=False)
        if df["log-likelihood"][0] > max_log_likelihood:
            max_log_likelihood = df["log-likelihood"][0]
            max_beta = df["beta"][0]
            max_batch = batch
        if np.isnan(df["log-likelihood"][0]) or -np.Inf == df["log-likelihood"][0]:
            print(f"invalid likelihood batch: {batch}\n")
    return max_beta, max_log_likelihood


def generate_spl_per_type_param_file(smi_range, beta_range, smi_res, beta_res, file_name):
    smis = np.arange(smi_range[0], smi_range[1], (smi_range[1] - smi_range[0]) / smi_res)
    betas = np.arange(beta_range[0], beta_range[1], (beta_range[1] - beta_range[0]) / beta_res)
    params = np.zeros((smi_res * beta_res, 2))
    for smi_idx in range(smi_res):
        for beta_idx in range(beta_res):
            params[smi_idx * beta_res + beta_idx, 0] = smis[smi_idx]
            params[smi_idx * beta_res + beta_idx, 1] = betas[beta_idx]
    with open(file_name, 'wb') as f:
        pickle.dump(params, f)


def detect_run_time_limit_termination(logs_path, job_array_id):
    file_list = os.listdir(logs_path)
    killed_indices = []
    for file in file_list:
        with open(os.path.join(logs_path, file), 'r') as f:
            content = f.read()
        if "TERM_RUNLIMIT" in content:
            job_arr_id_start = file.find(str(job_array_id))
            job_arr_id_end = job_arr_id_start + len(str(job_array_id))
            job_id_start = job_arr_id_end + 1
            job_id_end = file.find(".log")
            job_id = int(file[job_id_start: job_id_end])
            killed_indices.append(job_id)
    print(sorted(killed_indices))


def detect_no_space_on_node_error(logs_path, job_array_ids):
    file_list = os.listdir(logs_path)
    killed_indices = []
    for file in file_list:
        is_relevant_job = False
        for job_array_id in job_array_ids:
            if str(job_array_id) in file or 'error' not in file:
                is_relevant_job = True
                break
        if not is_relevant_job:
            continue
        with open(os.path.join(logs_path, file), 'r') as f:
            content = f.read()
        if "/scratch" in content:
            job_arr_id_start = file.find(str(job_array_id))
            job_arr_id_end = job_arr_id_start + len(str(job_array_id))
            job_id_start = job_arr_id_end + 1
            job_id_end = file.find(".error.log")
            job_id = int(file[job_id_start: job_id_end])
            killed_indices.append(job_id)
    print(sorted(killed_indices))
    print(len(killed_indices))
