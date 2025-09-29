from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse
import pickle
import copy
from contrastive_musicbert.utils.metadata_handler import MetadataHandler
from contrastive_musicbert.utils.retrieval_metrics import *
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

def get_all_similarties(
    ref_dir, sim_file="filtered_similarities_with_queries.pkl"
):
    sim_path = ref_dir / sim_file
    if sim_path.exists():
        with sim_path.open("rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"similarity file not found: {sim_path}")

def calculate_precision_and_recall_micro(df, thres, avc):
    all_version_count_meta = copy.deepcopy(avc)
    all_df = copy.deepcopy(df)

    mask = all_df >= thres
    columns_with_values_above_95 = mask.apply(
        lambda row: list(all_df.columns[row]), axis=1
    )
    result_df = columns_with_values_above_95.to_frame(name="Columns with Values > 0.95")

    cnt = 0
    invalid_errors = 0

    tp = 0
    fn = 0
    fp = 0
    total_dup = 0

    for index in tqdm(result_df.index):
        n_dup = all_version_count_meta[index]["count"]
        if n_dup == 1:
            continue

        total_dup += n_dup
        target = all_version_count_meta[index]["versions"]
        pred = result_df.loc[index].values[0]
        # since the target and pred already contains index itself, we can remove that
        try:
            target.remove(index)
            pred.remove(index)
        except Exception as e:
            invalid_errors += 1
            continue

        current_tp = len(set(target) & set(pred))
        current_fn = len(set(target) - set(pred))
        current_fp = len(set(pred) - set(target))

        tp += current_tp
        fn += current_fn
        fp += current_fp

        if current_tp == 0 and current_fp == 0:
            cnt += 1
            continue

    print(f"found {tp} duplicates from {total_dup}")
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if precision + recall == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, F1, cnt, invalid_errors

def calculate_precision_and_recall_macro(df, thres, avc, mode="ours"):
    all_version_count_meta = copy.deepcopy(avc)
    all_df = copy.deepcopy(df)
    mask = all_df >= thres
    columns_with_values_above_95 = mask.apply(
        lambda row: list(all_df.columns[row]), axis=1
    )
    result_df = columns_with_values_above_95.to_frame(name="Columns with Values > 0.95")

    cnt = 0
    invalid_errors = 0

    all_precisions = []
    all_recalls = []

    for index in result_df.index:
        n_dup = all_version_count_meta[index]["count"]
        if n_dup == 1:
            continue
        target = copy.deepcopy(all_version_count_meta[index]["versions"])
        pred = copy.deepcopy(result_df.loc[index].values[0])
        # since the target and pred already contains index itself, we can remove that
        try:
            target.remove(index)
            if index in pred:
                pred.remove(index)
        except:
            print("no identical item in retrieval error: this is not possible.")
            print(index)
            print(target)
            print(pred)
            # print(target[0])
            print(pred[0])
            print(df)
            break
            invalid_errors += 1
            continue

        # print(f"n_dup: {n_dup}, target: {target}, pred: {pred}")

        # calculate precision and recall
        tp = len(set(target) & set(pred))  # the model found the correct duplicates
        fn = len(
            set(target) - set(pred)
        )  # the model failed to find the correct duplicates
        fp = len(
            set(pred) - set(target)
        )  # the model found the incorrect duplicates

        recall = tp / (tp + fn)
        all_recalls.append(recall)

        if tp == 0 and fp == 0:
            cnt += 1
            continue

        precision = tp / (tp + fp)
        all_precisions.append(precision)

    # average precision and recall
    if len(all_precisions) == 0:
        average_precision = 0
    else:
        average_precision = sum(all_precisions) / len(all_precisions)
    if len(all_recalls) == 0:
        average_recall = 0
    else:
        average_recall = sum(all_recalls) / len(all_recalls)
    if (average_precision + average_recall) == 0:
        F1 = 0
    else:
        F1 = (
            2
            * (average_precision * average_recall)
            / (average_precision + average_recall)
        )
    return average_precision, average_recall, F1, cnt, invalid_errors


def calculate_precision_and_recall_all_thresolds(
    all_df,
    all_version_count_meta,
    thresholds=[0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99],
    mode="macro",
):
    aps = []
    ars = []
    F1s = []
    cnts = []
    invalid_errors_list = []

    for thres in thresholds:
        if mode == "micro":
            average_precision, average_recall, F1, cnt, invalid_errors = (
                calculate_precision_and_recall_micro(
                    all_df, thres, all_version_count_meta
                )
            )
        elif mode == "macro":
            average_precision, average_recall, F1, cnt, invalid_errors = (
                calculate_precision_and_recall_macro(
                    all_df, thres, all_version_count_meta
                )
            )
        print(
            f"thres: {thres}, average_precision: {average_precision:.3f}, F1: {F1:.3f}, average_recall: {average_recall:.3f}, cnt: {cnt}",
            f"invalid_errors: {invalid_errors}"
        )
        aps.append(average_precision)
        ars.append(average_recall)
        F1s.append(F1)
        cnts.append(cnt)
        invalid_errors_list.append(invalid_errors)

    return aps, ars, F1s, cnts, invalid_errors_list


def calculate_nDCG(filtered_row, target, k):
    all_indices = filtered_row.index.tolist()
    all_values = filtered_row.values.tolist()

    correct_answer_list, retrieval_binary_list = create_binary_list(target, all_indices)
    assert sum(correct_answer_list) == sum(
        retrieval_binary_list
    ), f"sum(correct_answer_list): {sum(correct_answer_list)}, sum(retrieval_binary_list): {sum(retrieval_binary_list)}"

    similarities = all_values
    ndcg = round(ndcg_score(retrieval_binary_list, similarities, k=k), 3)

    return ndcg


def check_NaN(metrics):
    for metric, value in metrics.items():
        if (
            isinstance(value, float) and not value == value
        ):  # Checking for NaN (NaN is not equal to itself)
            raise ValueError(f"NaN value detected for {metric}: {value}")


def calculate_MRR(filtered_row, target):
    top_k_values = filtered_row.nlargest(len(filtered_row))  # among all values
    top_indices = top_k_values.index.tolist()
    top_values = top_k_values.values.tolist()
    similarities = top_values

    correct_answer_list, retrieval_binary_list = create_binary_list(target, top_indices)
    mrr_value = round(mrr(retrieval_binary_list), 3)
    return mrr_value


def save_retrieval_metrics(metrics, ref_dir):
    ref_dir.mkdir(parents=True, exist_ok=True)
    file_path = ref_dir / "metrics_all.json"

    with open(file_path, "w") as json_file:
        json.dump(metrics, json_file, indent=4) 

    print(f"All retrieval metrics saved to {file_path}")

    metric_values = {metric: [] for metric in metrics[next(iter(metrics))].keys()}

    for index, metric_values_at_index in metrics.items():
        for metric, value in metric_values_at_index.items():
            metric_values[metric].append(value)

    file_path = ref_dir / "metrics_mean_std.json"
    metrics_stats = {}

    for metric, values in metric_values.items():
        metrics_stats[metric] = {"mean": np.mean(values), "std": np.std(values)}

    for metric, stats in metrics_stats.items():
        print(f"{metric}: Mean = {stats['mean']:.4f}, Std = {stats['std']:.4f}")

    with open(file_path, "w") as json_file:
        json.dump(
            metrics_stats, json_file, indent=4
        ) 

    print(f"Retrieval metric statistics saved to {file_path}")

    # 각 메트릭 별로 violin plot 그리기
    plt.figure(figsize=(15, 10))

    for i, (metric, values) in enumerate(metric_values.items(), 1):
        plt.subplot(2, 5, i)
        plt.violinplot(values)
        plt.title(metric)

    plt.tight_layout()

    # 그래프를 파일로 저장
    plot_filename = ref_dir / "metrics.png"
    plt.savefig(plot_filename)

    print(f"Retrieval metrics plot saved to {plot_filename}")

    # 각 메트릭 별로 box plot 그리기
    plt.figure(figsize=(15, 10))

    # 메트릭 이름과 해당 값들을 저장할 리스트 초기화
    metric_names = []
    metric_data = []

    # 각 메트릭 별로 데이터 수집
    for metric, values in metric_values.items():
        metric_names.append(metric)
        metric_data.append(values)

    # 박스 플롯 그리기
    plt.boxplot(metric_data, labels=metric_names)

    # 그래프 제목 및 축 이름 설정
    plt.title("Box Plot of Metrics")
    plt.ylabel("Values")

    # 그래프 보여주기
    plt.xticks(rotation=45)  # 메트릭 이름이 길 경우 축 레이블 회전
    plt.grid(True)
    plt.tight_layout()

    # 그래프를 파일로 저장
    plot_filename = ref_dir / "boxplot_metrics_plt.png"
    plt.savefig(plot_filename)

    print(f"Retrieval boxplot saved to {plot_filename}")

    return metrics_stats


def calculate_retrieval_metrics(df, all_version_count_meta):
    metrics = dict()
    for index, row in tqdm(df.iterrows()):
        if index not in df.columns:
            print(
                f"index: {index} not in queries. It is due to the filtering process. Skipping..."
            )
            continue
        filtered_row = row[[index != col for col in df.columns]]
        index = index.replace('.xml', '.1.xml') if '.1.xml' not in index else index
        target = copy.deepcopy(all_version_count_meta[index]["versions"])

        target.remove(index)  # remove the query itself

        target = [
            t for t in target if t in df.columns
        ]  # since the refs are filtered, we need to filter the target as well

        if len(target) == 0:
            print(
                "No refs to be compared exists. It is due the filtering process. Skipping..."
            )
            continue

        current_metric = dict()
        prev_nDCG = None

        for temp_k in [len(df.columns)]:
            ndcg = calculate_nDCG(filtered_row, target, temp_k)
            current_metric[f"nDCG@{temp_k}"] = ndcg
            if prev_nDCG is not None:
                assert prev_nDCG <= ndcg, f"prev_nDCG: {prev_nDCG}, ndcg: {ndcg}"
            prev_nDCG = ndcg

        current_metric["MRR"] = calculate_MRR(filtered_row, target)
        check_NaN(current_metric)

        metrics[index] = current_metric

    return metrics



def load_all_similarities(all_model_names, all_model_dirs):
    dfs = []
    for model_name, ref_dir in zip(all_model_names, all_model_dirs):
        print("load the condition: ", model_name)
        all_df = get_all_similarties(ref_dir)
        dfs.append(all_df)

    return dfs

@hydra.main(version_base=None, config_path="./yamls/", config_name="evaluation")
def main(config: DictConfig):
    if not config.cal_retrieval and not config.cal_classification:
        raise NotImplementedError('you should select either --cal_retrieval or --cal_classification')

    with open(config.eval_models_path, "r") as f:
        model_dict = json.load(f)

    all_model_names = list(model_dict.keys())
    all_model_dirs = list(model_dict.values())
    all_model_dirs = [Path(config.eval_dir)/model for model in all_model_dirs]

    metadata_handler = MetadataHandler()
    all_version_count_meta = metadata_handler.get_all_version_count_meta()
    assert 'Billy Joel__Honesty.2' in all_version_count_meta.keys(), "Billy Joel__Honesty.2 should be in the dataset"

    dfs = load_all_similarities(all_model_names, all_model_dirs)
    
    # assert all dfs has same shape
    for i in range(1, len(dfs)):
        assert dfs[0].shape == dfs[i].shape, "all dfs should have same shape"
    print(dfs[0].shape)

    if config.cal_retrieval:
        for model_name, all_df, ref_dir in tqdm(
            zip(all_model_names, dfs, all_model_dirs), total=len(all_model_names)
        ):
            print('calculating retrieval metrics...')
            metrics = calculate_retrieval_metrics(all_df, all_version_count_meta)
            save_retrieval_metrics(metrics, ref_dir / "retrieval_metrics_details")

    if config.cal_classification:
        for model_name, all_df, ref_dir in tqdm(
            zip(all_model_names, dfs, all_model_dirs), total=len(all_model_names)
        ):
            print("evaluating the condition: ", model_name)
            aps, ars, F1s, cnts, invalid_errors_list = (
                calculate_precision_and_recall_all_thresolds(
                    all_df, all_version_count_meta, thresholds=config.thres, mode="macro"
                )
            )

            output_file = ref_dir / "classification_metrics.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"Classification metrics saved to {output_file}")
            df = pd.DataFrame(
                {
                    "thres": config.thres,
                    "aps": aps,
                    "ars": ars,
                    "F1s": F1s,
                    "cnts": cnts,
                    "invalid_errors_list": invalid_errors_list,
                }
            )
            df.to_csv(output_file, index=True)

if __name__ == "__main__":
    main()