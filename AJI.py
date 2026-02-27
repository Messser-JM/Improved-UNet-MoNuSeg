from skimage.measure import label
import numpy as np
def calculate_aji(gt_mask, pred_mask):
    """
    Calculate the AJI of a single image (Aggregated Jaccard Index)
    gt_mask: 0/1 binary numpy array
    pred_mask: 0/1 binary numpy array
    """
    gt_labels = label(gt_mask)
    pred_labels = label(pred_mask)

    gt_ids = np.unique(gt_labels)[1:]
    pred_ids = np.unique(pred_labels)[1:]

    overall_inter = 0
    overall_union = 0
    pred_used = set()

    for gid in gt_ids:
        gt_obj_mask = (gt_labels == gid)
        gt_area = np.sum(gt_obj_mask)

        intersecting_preds = np.unique(pred_labels[gt_obj_mask])
        intersecting_preds = intersecting_preds[intersecting_preds != 0]

        if len(intersecting_preds) == 0:
            overall_union += gt_area
            continue

        best_iou = 0
        best_pred_id = 0

        for pid in intersecting_preds:
            pred_obj_mask = (pred_labels == pid)
            intersection = np.sum(gt_obj_mask & pred_obj_mask)
            union = np.sum(gt_obj_mask | pred_obj_mask)
            iou = intersection / union

            if iou > best_iou:
                best_iou = iou
                best_pred_id = pid

        if best_pred_id != 0:
            pred_obj_mask = (pred_labels == best_pred_id)
            intersection = np.sum(gt_obj_mask & pred_obj_mask)
            union = np.sum(gt_obj_mask | pred_obj_mask)
            overall_inter += intersection
            overall_union += union
            pred_used.add(best_pred_id)
        else:
            overall_union += gt_area

    for pid in pred_ids:
        if pid not in pred_used:
            overall_union += np.sum(pred_labels == pid)

    if overall_union == 0: return 1.0
    return overall_inter / overall_union


def batch_aji(preds_tensor, masks_tensor):
    """Batch AJI wrapper"""
    bs = preds_tensor.shape[0]
    preds_np = (preds_tensor > 0.5).detach().cpu().numpy().astype(np.uint8)
    masks_np = masks_tensor.detach().cpu().numpy().astype(np.uint8)

    total_aji = 0
    for i in range(bs):
        total_aji += calculate_aji(masks_np[i, 0], preds_np[i, 0])
    return total_aji
