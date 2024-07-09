import os
import numpy as np
from PIL import Image
from metrics import iou_score
import pandas as pd

GLOBAL_PATH = '../CloudNet/Full_Cloud/'
TEST_PATH = os.path.join(GLOBAL_PATH, 'test/')


def load_mask_couple(mask_tuple):


    gt_image = Image.open(mask_tuple[0])
    gt_image.save('gt_mask.TIF')

    pred_image = Image.open(mask_tuple[1])
    pred_image.save('pred_mask.TIF')

    gt_mask = np.array(gt_image)
    pred_mask = np.array(pred_image)
  

    return gt_mask, pred_mask

def run_accuracy(gt_mask, pred_mask):


    # return a datastructure with the metrics

    iou = iou_score(gt_mask, pred_mask)

    metrics = {
        "iou": iou
    }

    return metrics



def run_accuracy_on_couples_path(coupled_masks_path):
    """
    Gather metrics and get a mean value by scene and overall and save it in a csv file.
    One file for each scene with the mean and std of all the metrics computed by run_accuracy,
    and one file with the overall metrics.
    """
    os.makedirs('metrics', exist_ok=True)
    scene_metrics = {}

    for scene, mask_tuples in coupled_masks_path.items():
        metrics_list = []  # List to store metrics dictionaries for each mask

        for mask_tuple in mask_tuples:
            gt_mask, pred_mask = load_mask_couple(mask_tuple)
            metrics_dict = run_accuracy(gt_mask, pred_mask)
            metrics_list.append(metrics_dict)  # Append the metrics dictionary to the list

        # Convert the list of dictionaries to a DataFrame and calculate mean and std
        scene_metrics_df = pd.DataFrame(metrics_list)
        scene_metrics[scene] = {
            'mean': scene_metrics_df.mean().to_dict(),
            'std': scene_metrics_df.std().to_dict()
        }

        # Save the scene metrics in a csv file in the metrics folder
        scene_metrics_df.describe().to_csv(f'metrics/{scene}_metrics.csv')

    # To calculate overall metrics, first flatten the list of all metrics dictionaries
    all_metrics_list = []
    for metrics in scene_metrics.values():
        all_metrics_list.extend([metrics['mean'], metrics['std']])  # Extend the list with mean and std dictionaries

    overall_metrics_df = pd.DataFrame(all_metrics_list)
    overall_mean = overall_metrics_df.mean().to_dict()
    overall_std = overall_metrics_df.std().to_dict()
    overall_metrics = {'mean': overall_mean, 'std': overall_std}

    # Save the overall metrics in a csv file
    pd.DataFrame([overall_metrics]).to_csv('metrics/overall_metrics.csv')







def create_coupled_path():
    # TODO for each folder in predictions/ iterate through the masks and find the corresponding mask in the gt folder
    # Create tuple for each mask and gt mask and save the scene it belongs to
    # Calculate the accuracy metrics 
    # Save the metrics in a csv file

    coupled_masks_path = {}

    # iterate through the predictions folder
    for scene in os.listdir('predictions'):
    # Create an entry in a directory for the scene
        coupled_masks_path[scene] = []

        # iterate through the masks in the folder
        for mask in os.listdir(os.path.join('predictions', scene, 'mask')):
            predicted_mask_path = os.path.join('predictions', scene, 'mask', mask)
            # Assuming TEST_PATH is defined elsewhere
            gt_mask_path = os.path.join(TEST_PATH, 'test_gt', f'gt_{mask}')
            coupled_masks_path[scene].append(( gt_mask_path, predicted_mask_path))


    return coupled_masks_path
