import os
import numpy as np
from PIL import Image
import pandas as pd
from utils.config import Config
from accuracy.metrics import jaccard_coefficient, dice_coefficient

def load_mask_couple(mask_tuple):
    gt_image = Image.open(mask_tuple[0])
    pred_image = Image.open(mask_tuple[1])
    gt_mask = np.array(gt_image).astype(bool)
    pred_mask = np.array(pred_image).astype(bool)
    return gt_mask, pred_mask

def run_accuracy(gt_mask, pred_mask):
    metrics = {
        "jacc": jaccard_coefficient(gt_mask, pred_mask),
        "dice": dice_coefficient(gt_mask, pred_mask),
        # Add more metrics here as needed
    }
    return metrics

def calculate_percentile_metrics(df, percentiles=[90, 99]):
    percentile_metrics = {}
    for metric in df.columns:
        for percentile in percentiles:
            threshold = np.percentile(df[metric], percentile)
            percentile_metrics[f'top_{percentile}%_{metric}'] = df[df[metric] >= threshold][metric].mean()
    return percentile_metrics

def run_accuracy_on_couples_path(coupled_masks_path, output_dir=Config.METRICS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    scene_metrics = {}
    all_metrics_list = []

    for scene, mask_tuples in coupled_masks_path.items():
        metrics_list = []

        for mask_tuple in mask_tuples:
            gt_mask, pred_mask = load_mask_couple(mask_tuple)
            metrics_dict = run_accuracy(gt_mask, pred_mask)
            metrics_list.append(metrics_dict)

        scene_metrics_df = pd.DataFrame(metrics_list)
        scene_metrics[scene] = {
            'mean': scene_metrics_df.mean().to_dict(),
            'std': scene_metrics_df.std().to_dict()
        }
        
        # Calculate top 90% and 99% metrics for all metrics
        percentile_metrics = calculate_percentile_metrics(scene_metrics_df)
        scene_metrics[scene].update(percentile_metrics)

        # Prepare data for CSV output
        output_data = {'metric': ['mean', 'std'] + [f'top_{p}%' for p in [90, 99]]}
        for metric in scene_metrics_df.columns:
            output_data[metric] = [
                scene_metrics[scene]['mean'][metric],
                scene_metrics[scene]['std'][metric],
                scene_metrics[scene][f'top_90%_{metric}'],
                scene_metrics[scene][f'top_99%_{metric}']
            ]

        # Save the scene metrics in a csv file in the metrics folder
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(os.path.join(output_dir, f'{scene}_metrics.csv'), index=False)

        all_metrics_list.extend(metrics_list)

    # Calculate overall metrics
    overall_metrics_df = pd.DataFrame(all_metrics_list)
    overall_mean = overall_metrics_df.mean().to_dict()
    overall_std = overall_metrics_df.std().to_dict()

    # Calculate top 90% and 99% for overall metrics
    overall_percentile_metrics = calculate_percentile_metrics(overall_metrics_df)

    # Prepare data for overall CSV output
    overall_output_data = {'metric': ['mean', 'std'] + [f'top_{p}%' for p in [90, 99]]}
    for metric in overall_metrics_df.columns:
        overall_output_data[metric] = [
            overall_mean[metric],
            overall_std[metric],
            overall_percentile_metrics[f'top_90%_{metric}'],
            overall_percentile_metrics[f'top_99%_{metric}']
        ]

    # Save the overall metrics in a csv file
    overall_output_df = pd.DataFrame(overall_output_data)
    overall_output_df.to_csv(os.path.join(output_dir, 'overall_metrics.csv'), index=False)

def create_coupled_path(path=Config.PREDICTION_PATH):
    coupled_masks_path = {}
    for scene in os.listdir(path):
        coupled_masks_path[scene] = []
        for mask in os.listdir(os.path.join(Config.PREDICTION_PATH, scene, Config.MASK)):
            predicted_mask_path = os.path.join(Config.PREDICTION_PATH, scene, Config.MASK, mask)
            gt_mask_path = os.path.join(Config.TEST_PATH, Config.TEST_GT_DIR, f'{Config.GT_PREFIX}{mask}')
            coupled_masks_path[scene].append((gt_mask_path, predicted_mask_path))
    return coupled_masks_path