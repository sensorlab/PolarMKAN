import wandb
import pandas as pd
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

api = wandb.Api()

def prepare_data_from_wandb(
    epoch: int,
    metrics: list[str],
    user_name: str = 'mkhlkrasnov-i-am',
    project: str = 'WiSig_evaluations_for_paper_same_days',
    grouping_keys : list[str] = ['approach__name', 'exp_id']
) -> tuple[pd.DataFrame, list[str]]:
    """
    Fetch run data at a specific epoch and return a cleaned DataFrame with metadata.
    
    Args:
        epoch (int): index of the epoch to calculate stats. 
        metrics (list[str]): list of metrics to calculate mde.
        user_name (str): user name from wandb.
        project (str): project name with test run for mde.
        grouping_keys (list[str]): list of keys in config to group raw logs.
    Returns:
        Prepared DataFrame
    """
    
    runs = api.runs(f"{user_name}/{project}")
    all_metrics = []
    
    for run in runs:
        df = run.history()
        config = run.config
        
        if not df.empty:
            cur_df = df[df["_step"] == epoch][metrics + ['_runtime']].copy()
            for key in grouping_keys:
                key_splitted = key.split('__')
                cur_lvl_config = config
                for cur_key_lvl in key_splitted:
                    cur_lvl_config = cur_lvl_config.get(cur_key_lvl, None)
                    if cur_lvl_config is None:
                        cur_lvl_config = 'none'
                        break
                cur_df[key] = cur_lvl_config      
            all_metrics.append(cur_df)

    combined_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    
    return combined_df


def mde(
    data: pd.DataFrame,
    num_iterations: int = 1,
    alpha: float = 0.05,
    beta: float = 0.2
) -> pd.DataFrame:
    """Calculate Minimum Detectable Effect (MDE) for metrics over several iterations."""
    
    t_alpha = norm.ppf(1 - alpha / 2)
    t_beta = norm.ppf(1 - beta)
    n = len(data)

    results = []

    for i in range(1, num_iterations + 1):
        stats = {}
        for metric in data.columns:
            std = data[metric].std()
            mean = data[metric].mean()
            mde_val = (t_beta + t_alpha) * std / np.sqrt(n * i / 2)
            
            stats[f"{metric}_{i}_mde"] = mde_val
            stats[f"{metric}_{i}_mean"] = mean
            stats[f"{metric}_{i}_mde_rel"] = mde_val / mean * 100 if mean != 0 else np.nan
        
        stats['_runtime_mean'] = data['_runtime'].mean()
        stats['len_data'] = data.shape[0]
        
        results.append(pd.DataFrame([stats]))

    return pd.concat(results, axis=1)


def calculate_mde_from_runs(
    data: pd.DataFrame,
    grouping_keys: list[str],
    num_iterations = 1
) -> pd.DataFrame:
    """Group data and apply MDE calculation per group."""
    mde_func = lambda data: mde(data, num_iterations = num_iterations) 
    return data.groupby(by=grouping_keys).apply(mde_func)

def draw_mde_curve(mdes_dataframe, metric, num_iterations, type = 'rel'):
    for index, row in mdes_dataframe.iterrows():
        if type == 'rel':
            columns = [f"{metric}_{i}_mde_rel" for i in range(1, num_iterations + 1)]
        elif type == 'abs':
            columns = [f"{metric}_{i}_mde" for i in range(1, num_iterations + 1)]
        values = row[columns]

        label = ''
        for key in index[:-1]:
            if key != 'none':
                label += f"{key}."
        plt.plot(
            [i+1 for i in range(num_iterations)],
            values,
            label = label
        )
    plt.legend()
    plt.xlabel('num_iterations')
    plt.ylabel(f'mde_{type}')

def draw_time_iter_curve(mdes_dataframe, num_iterations):
    for index, row in mdes_dataframe.iterrows():
        values = row['_runtime_mean'].values[0] / 60
        values *= row['len_data'].values[0]
        
        label = ''
        for key in index[:-1]:
            if key != 'none':
                label += f"{key}."
        plt.plot(
            [i+1 for i in range(num_iterations)],
            [values * (i+1) for i in range(num_iterations)],
            label = label
        )
    plt.legend()
    plt.xlabel('num_iterations')
    plt.ylabel('runtime, minutes')