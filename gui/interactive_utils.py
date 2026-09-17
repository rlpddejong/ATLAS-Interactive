# Modified from https://github.com/seoungwugoh/ivs-demo

from typing import Literal, List
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from gui.cutie.utils.palette import custom_palette


def image_to_torch(frame: np.ndarray, device: str = 'cuda'):
    # frame: H*W*3 numpy array
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float().to(device, non_blocking=True) / 255
    return frame


def torch_prob_to_numpy_mask(prob: torch.Tensor):
    mask = torch.max(prob, dim=0).indices
    mask = mask.cpu().numpy().astype(np.uint8)
    return mask


def index_numpy_to_one_hot_torch(mask: np.ndarray, num_classes: int):
    mask = torch.from_numpy(mask).long()
    return F.one_hot(mask, num_classes=num_classes).permute(2, 0, 1).float()


"""
Some constants fro visualization
"""
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
except:
    device = torch.device("cpu")

color_map_np = np.frombuffer(custom_palette, dtype=np.uint8).reshape(-1, 3).copy()
# scales for better visualization
color_map_np = (color_map_np.astype(np.float32) * 1.5).clip(0, 255).astype(np.uint8)
color_map = color_map_np.tolist()
color_map_torch = torch.from_numpy(color_map_np).to(device) / 255



def get_visualization(mode: Literal['image', 'mask', 'overlay', 'overlay+numbers'],
                      image: np.ndarray,
                      mask: np.ndarray,
                      layer: np.ndarray,
                      target_objects: List[int]) -> np.ndarray:
    if mode == 'image':
        return image
    elif mode == 'mask':
        return color_map_np[mask]
    elif mode == 'overlay':
        return overlay_davis(image, mask)
    elif mode == 'overlay+numbers':
        return draw_class_numbers(overlay_davis(image, mask), mask)
    else:
        raise NotImplementedError


def get_visualization_torch(mode: Literal['image', 'mask', 'overlay', 'overlay+numbers'],
                            image: torch.Tensor,
                            prob: torch.Tensor,
                            layer: torch.Tensor,
                            target_objects: List[int]) -> np.ndarray:
    if mode == 'image':
        return image
    elif mode == 'mask':
        mask = torch.max(prob, dim=0).indices
        return (color_map_torch[mask] * 255).byte().cpu().numpy()
    elif mode == 'overlay':
        return overlay_davis_torch(image, prob)
    elif mode == 'overlay+numbers':
        mask = torch.max(prob, dim=0).indices.cpu().numpy()
        return draw_class_numbers(overlay_davis_torch(image, prob), mask)
    else:
        raise NotImplementedError


def draw_class_numbers(image: np.ndarray, mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """ Draw each class's numeric id at the centroid of its connected components. """
    result = np.ascontiguousarray(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    for class_id in np.unique(mask):
        if class_id == 0:
            continue
        binary_mask = (mask == class_id).astype(np.uint8)
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary_mask,
                                                                            connectivity=8)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] < min_area:
                continue
            cx, cy = centroids[label]
            text = str(int(class_id))
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 2)
            org = (int(cx - text_w / 2), int(cy + text_h / 2))
            # black outline for legibility on top of any color
            cv2.putText(result, text, org, font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(result, text, org, font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def overlay_davis(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, fade: bool = False):
    """ Overlay segmentation on top of RGB image. from davis official"""
    im_overlay = image.copy()

    colored_mask = color_map_np[mask]
    foreground = image * alpha + (1 - alpha) * colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    if fade:
        im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6
    return im_overlay.astype(image.dtype)


def overlay_davis_torch(image: torch.Tensor,
                        prob: torch.Tensor,
                        alpha: float = 0.5,
                        fade: bool = False):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # Changes the image in-place to avoid copying
    # NOTE: Make sure you no longer use image after calling this function
    image = image.permute(1, 2, 0)
    im_overlay = image
    mask = torch.max(prob, dim=0).indices

    colored_mask = color_map_torch[mask]
    foreground = image * alpha + (1 - alpha) * colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    if fade:
        im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6

    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay




