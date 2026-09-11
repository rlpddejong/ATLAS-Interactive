import os
from os import path
import logging
from typing import List, Literal

import cv2
# fix conflicts between qt5 and cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

import torch
try:
    from torch import mps
except:
    print('torch.MPS not available.')
from torch import autocast
from torchvision.transforms.functional import to_tensor
import numpy as np
from omegaconf import DictConfig, open_dict

from gui.cutie.model.cutie import CUTIE
from gui.cutie.inference.inference_core import InferenceCore

from gui.interaction import *
from gui.interactive_utils import *
from gui.resource_manager import ResourceManager, resolve_workspace
from gui.gui import GUI
from gui.click_controller import ClickController
from gui.reader import PropagationReader, get_data_loader
from gui.exporter import convert_frames_to_video, convert_mask_to_binary
from gui.cutie.utils.download_models import download_models_if_needed
from gui.source_dialog import prompt_for_source
from gui.history_logger import HistoryLogger

from gui.cutie.utils.palette import custom_palette_np # added

log = logging.getLogger()



class MainController():

    def __init__(self, cfg: DictConfig, status_callback=None) -> None:
        super().__init__()

        self.initialized = False
        # optional callback(str | None) used to report startup progress, e.g. to a
        # splash screen; called with None once the GUI is ready to be shown.
        self._status_callback = status_callback

        # setting up the workspace
        resolve_workspace(cfg)

        # reading arguments
        self.cfg = cfg
        self.num_objects = cfg['num_objects']
        self.device = cfg['device']
        self.amp = cfg['amp']

        # initializing the network(s)
        self._report_status('Loading segmentation models...')
        self.initialize_networks()

        # main components
        self._report_status('Preparing workspace (extracting frames if needed)...')
        self.res_man = ResourceManager(cfg)
        self.history = HistoryLogger(cfg['workspace'])
        if 'workspace_init_only' in cfg and cfg['workspace_init_only']:
            self._report_status(None)
            return
        self.processor = InferenceCore(self.cutie, self.cfg)
        self.gui = GUI(self, self.cfg)

        # initialize control info
        self.length: int = self.res_man.length
        self.interaction: Interaction = None
        self.interaction_type: str = 'Click'
        self.curr_ti: int = 0
        self.curr_object: int = 1
        self.locked_object_ids: set = set()
        self.propagating: bool = False
        self.propagate_direction: Literal['forward', 'backward', 'none'] = 'none'
        self.last_ex = self.last_ey = 0

        # current frame info
        self.curr_frame_dirty: bool = False
        self.curr_image_np: np.ndarray = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.curr_image_torch: torch.Tensor = None
        self.curr_mask: np.ndarray = np.zeros((self.h, self.w), dtype=np.uint8)
        self.curr_prob: torch.Tensor = torch.zeros((self.num_objects + 1, self.h, self.w),
                                                   dtype=torch.float).to(self.device)
        self.curr_prob[0] = 1

        # visualization info
        self.vis_mode: str = 'overlay'
        self.vis_image: np.ndarray = None
        self.save_visualization_mode: str = 'None'
        self.save_soft_mask: bool = False

        self.interacted_prob: torch.Tensor = None
        self.overlay_layer: np.ndarray = None
        self.overlay_layer_torch: torch.Tensor = None

        # the object id used for popup/layer overlay
        self.vis_target_objects = list(range(1, self.num_objects + 1))

        self.load_current_image_mask()
        self.show_current_frame()

        # initialize stuff
        self.update_memory_gauges()
        self.update_gpu_gauges()
        self.gui.work_mem_min.setValue(self.processor.memory.min_mem_frames)
        self.gui.work_mem_max.setValue(self.processor.memory.max_mem_frames)
        self.gui.long_mem_max.setValue(self.processor.memory.max_long_tokens)
        self.gui.mem_every_box.setValue(self.processor.mem_every)

        # for exporting videos
        self.output_fps = cfg['output_fps']
        self.output_bitrate = cfg['output_bitrate']

        # set callbacks
        self.gui.on_mouse_motion_xy = self.on_mouse_motion_xy
        self.gui.click_fn = self.click_fn

        # Variables for polygon drawing and hovering first point
        self.polygon_points = []
        self.hover_first_point = False
        self.hover_threshold = 8  # pixels
        self.in_polygon_mode = False

        self.gui.show()
        self._report_status(None)
        self.gui.text('Initialized.')
        self.initialized = True
        self.history.log('session_started', video=cfg['video'], images=cfg['images'])

        # try to load the default overlay
        self._try_load_layer('./docs/uiuc.png')
        self.gui.set_current_object_id(self.curr_object)
        self.update_config()

    def _report_status(self, message) -> None:
        if self._status_callback is not None:
            self._status_callback(message)

    def initialize_networks(self) -> None:
        download_models_if_needed()
        self.cutie = CUTIE(self.cfg).eval().to(self.device)
        model_weights = torch.load(self.cfg.weights, map_location=self.device)
        self.cutie.load_weights(model_weights)

        self.click_ctrl = ClickController(self.cfg.ritm_weights, device=self.device)

    def on_load_new_source(self):
        if self.propagating:
            self.gui.text('Please pause propagation before loading a new video.')
            return

        source = prompt_for_source()
        if source is None:
            return

        # persist any pending edit on the current frame before switching away from it
        if self.curr_frame_dirty:
            self.save_current_mask()
            self.curr_frame_dirty = False

        with open_dict(self.cfg):
            self.cfg['video'] = source.get('video')
            self.cfg['images'] = source.get('images')
            self.cfg['workspace'] = None

        self.gui.text('Loading new source, please wait...')
        self.gui.process_events()

        # reusing the already-loaded model weights; only the workspace/resources are swapped
        self.res_man = ResourceManager(self.cfg)
        self.processor = InferenceCore(self.cutie, self.cfg)
        self.history = HistoryLogger(self.cfg['workspace'])
        self.history.log('session_started', video=self.cfg['video'], images=self.cfg['images'])

        # reset per-video state
        self.length = self.res_man.length
        self.interaction = None
        self.curr_ti = 0
        self.locked_object_ids = set()
        self.propagating = False
        self.propagate_direction = 'none'
        self.curr_frame_dirty = False
        self.interacted_prob = None
        self.polygon_points = []
        self.hover_first_point = False
        self.in_polygon_mode = False

        self.curr_image_np = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.curr_image_torch = None
        self.curr_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.curr_prob = index_numpy_to_one_hot_torch(self.curr_mask, self.num_objects + 1).to(
            self.device)

        self.gui.reload_for_new_source()
        self.gui.pause_propagation()  # re-enable buttons/slider in case a run was interrupted

        self.load_current_image_mask()
        self.show_current_frame()

        self.update_memory_gauges()
        self.update_gpu_gauges()
        self.gui.reset_locks()
        self.gui.set_current_object_id(self.curr_object)

        self.gui.text(f'Loaded new source. Workspace: {self.cfg["workspace"]}')

    def hit_number_key(self, number: int):
        if number == self.curr_object:
            return
        self.curr_object = number
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()
        self.gui.text(f'Current object changed to {number}.')
        self.gui.set_current_object_id(number)
        self.show_current_frame()

    def on_toggle_lock(self, object_id: int):
        # a locked class can't be hand-edited, and during propagation its
        # mask is restored from whatever was already saved on disk for each
        # frame instead of being re-predicted by the model -- see
        # _restore_locked_classes() for how that's enforced
        if object_id in self.locked_object_ids:
            self.locked_object_ids.discard(object_id)
            self.gui.text(f'Class {object_id} unlocked.')
        else:
            self.locked_object_ids.add(object_id)
            self.gui.text(f'Class {object_id} locked.')
        self.history.log('toggle_lock',
                          object_id=object_id,
                          locked=(object_id in self.locked_object_ids))
        self.gui.set_lock_state(object_id, object_id in self.locked_object_ids)

    def on_lock_all(self):
        # toggles: locks everything, unless everything is already locked, in
        # which case it unlocks everything
        locking = len(self.locked_object_ids) < self.num_objects
        self.locked_object_ids = set(range(1, self.num_objects + 1)) if locking else set()
        for object_id in range(1, self.num_objects + 1):
            self.gui.set_lock_state(object_id, locking)
        self.gui.set_lock_all_button_state(locking)
        self.history.log('lock_all', locked=locking)
        self.gui.text('All classes locked.' if locking else 'All classes unlocked.')

    def on_mouse_motion_xy(self, x: int, y: int):
        # Check if polygon is being drawn and at least one point exists
        if self.polygon_points:
            # Check distance to first point
            first_pt = self.polygon_points[0]
            dist = ((x - first_pt[0])**2 + (y - first_pt[1])**2)**0.5
            was_hovering = self.hover_first_point
            self.hover_first_point = dist <= self.hover_threshold
            
            # If hover state changed, redraw polygon overlay to update color
            if self.hover_first_point != was_hovering:
                self.compose_polygon_overlay()
                self.update_canvas()
        self.last_ex, self.last_ey = x, y

    def compose_polygon_overlay(self):
        # Reset to base visualization image
        self.compose_current_im()

        # Draw polygon points and lines
        pts = [(int(px), int(py)) for (px, py) in self.polygon_points]

        # Get color for the current object
        r, g, b = custom_palette_np[self.curr_object] 
        r, g, b = int(r), int(g), int(b)

        # Draw lines between points
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                cv2.line(self.vis_image, pts[i], pts[i + 1], color=(r,g,b), thickness=1)

        # Draw points with hover effect on first point
        for i, pt in enumerate(pts):
            if i == 0 and self.hover_first_point:
                # Hover color: white
                color = (255, 255, 255)
                radius = 6
            else:
                # Normal color: yellow
                color = (r,g,b)
                radius = 4
            cv2.circle(self.vis_image, pt, radius=radius, color=color, thickness=-1)

    def click_fn(self, action: Literal['left', 'right', 'middle'], x: int, y: int):
        if self.propagating:
            return

        if not hasattr(self, 'in_polygon_mode'):
            self.in_polygon_mode = False  # new flag to track current mode

        if action == 'middle':
            # Toggle polygon mode
            self.in_polygon_mode = not self.in_polygon_mode
            self.polygon_points = []
            self.hover_first_point = False
            mode_text = 'Polygon mode ON' if self.in_polygon_mode else 'Click mode ON'
            self.gui.text(mode_text)
            self.compose_current_im()
            self.update_canvas()
            return

        if self.curr_object in self.locked_object_ids:
            self.gui.text(f'Class {self.curr_object} is locked. Unlock it to edit.')
            return

        if self.in_polygon_mode:
            # In polygon drawing mode
            if action == 'left':
                if self.polygon_points:
                    first_pt = self.polygon_points[0]
                    dist = ((x - first_pt[0])**2 + (y - first_pt[1])**2)**0.5
                    if dist <= self.hover_threshold:
                        # Finalize polygon
                        self.gui.text('Finalizing polygon.')

                        # Close polygon loop
                        if self.polygon_points[-1] != first_pt:
                            self.polygon_points.append(first_pt)

                        # Create binary mask
                        mask = np.zeros((self.h, self.w), dtype=np.uint8)
                        pts_np = np.array([[(int(px), int(py)) for px, py in self.polygon_points]], dtype=np.int32)
                        cv2.fillPoly(mask, pts_np, color=1)
                        self.curr_mask[mask > 0] = self.curr_object
                        self.save_current_mask()

                        # ✅ Update probability map so it's used for propagation
                        self.curr_prob = index_numpy_to_one_hot_torch(self.curr_mask, self.num_objects + 1).to(self.device)

                        self.history.log('polygon_completed',
                                          frame=self.curr_ti,
                                          object_id=self.curr_object,
                                          num_points=len(self.polygon_points))

                        self.polygon_points = []
                        self.hover_first_point = False
                        self.show_current_frame()
                        self.gui.text('Polygon finalized and added to segmentation.')
                        return
                # Add new point
                self.polygon_points.append((x, y))
                self.gui.text(f'Polygon point added: ({x}, {y})')
                self.history.log('polygon_point_added',
                                  frame=self.curr_ti,
                                  object_id=self.curr_object,
                                  point=[x, y])
                self.compose_polygon_overlay()
                self.update_canvas()
                return

            elif action == 'right':
                # Remove last point
                if self.polygon_points:
                    removed = self.polygon_points.pop()
                    self.gui.text(f'Removed polygon point: {removed}')
                    self.history.log('polygon_point_removed',
                                      frame=self.curr_ti,
                                      object_id=self.curr_object,
                                      point=list(removed))
                    self.compose_polygon_overlay()
                    self.update_canvas()
                else:
                    self.gui.text('No points to remove.')
                return

            else:
                # Do nothing in polygon mode for other actions
                return

        # Not in polygon mode: do normal click interaction
        last_interaction = self.interaction
        new_interaction = None

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            if action in ['left', 'right']:
                self.convert_current_image_mask_torch()
                image = self.curr_image_torch
                if (last_interaction is None or last_interaction.tar_obj != self.curr_object):
                    self.complete_interaction()
                    self.click_ctrl.unanchor()
                    new_interaction = ClickInteraction(image, self.curr_prob, (self.h, self.w),
                                                    self.click_ctrl, self.curr_object)
                    if new_interaction is not None:
                        self.interaction = new_interaction

                self.interaction.push_point(x, y, is_neg=(action == 'right'))
                self.interacted_prob = self.interaction.predict().to(self.device, non_blocking=True)
                self.update_interacted_mask()
                self.update_gpu_gauges()

                self.history.log('click',
                                  frame=self.curr_ti,
                                  object_id=self.curr_object,
                                  action=action,
                                  point=[x, y])

    def load_current_image_mask(self, no_mask: bool = False):
        self.curr_image_np = self.res_man.get_image(self.curr_ti)
        self.curr_image_torch = None

        if not no_mask:
            loaded_mask = self.res_man.get_mask(self.curr_ti)
            if loaded_mask is None:
                self.curr_mask.fill(0)
            else:
                self.curr_mask = loaded_mask.copy()
            self.curr_prob = None

    def convert_current_image_mask_torch(self, no_mask: bool = False):
        if self.curr_image_torch is None:
            self.curr_image_torch = to_tensor(self.curr_image_np).to(self.device, non_blocking=True)

        if self.curr_prob is None and not no_mask:
            self.curr_prob = index_numpy_to_one_hot_torch(self.curr_mask, self.num_objects + 1).to(
                self.device, non_blocking=True)

    def _present_object_ids(self) -> List[int]:
        # which of the num_objects classes actually have annotated pixels on
        # the current frame -- only these get fed to the model, so it only
        # has to internally track objects that are actually in use instead
        # of all num_objects classes regardless of whether they appear here
        #
        # ObjectManager.add_new_objects() requires its input ordered by each
        # object's internal tmp_id (assigned in first-annotated order), not
        # by real class id -- those two orders diverge once classes have been
        # drawn out of numeric order across different frames/sessions. Already
        # -tracked objects must come first, sorted by their existing tmp_id;
        # brand-new objects go after (any order among them is fine, since
        # add_new_objects() hands out fresh tmp_ids sequentially as it sees
        # them, which stays sorted appended after the tracked ones).
        ids = [int(i) for i in np.unique(self.curr_mask) if i != 0]
        obj_manager = self.processor.object_manager

        def sort_key(obj_id):
            if obj_id in obj_manager.obj_id_to_obj:
                return (0, obj_manager.find_tmp_by_id(obj_id))
            return (1, obj_id)

        return sorted(ids, key=sort_key)

    def _curr_mask_torch(self) -> torch.Tensor:
        return torch.from_numpy(self.curr_mask.astype(np.int64)).to(self.device, non_blocking=True)

    def _scatter_sparse_prob(self, sparse_prob: torch.Tensor) -> torch.Tensor:
        # the processor only tracks objects that have been annotated, indexed
        # by an internal "tmp_id" (assigned in the order objects were first
        # added) rather than by their real class id. Scatter its output back
        # into the dense num_objects+1 layout (channel i == class i) that the
        # rest of the GUI (visualization, click interaction, undo) expects.
        dense = torch.zeros((self.num_objects + 1, *sparse_prob.shape[1:]),
                            dtype=sparse_prob.dtype,
                            device=sparse_prob.device)
        dense[0] = sparse_prob[0]
        for tmp_id, obj in self.processor.object_manager.tmp_id_to_obj.items():
            dense[obj.id] = sparse_prob[tmp_id]
        return dense

    def _mask_from_sparse_prob(self, sparse_prob: torch.Tensor) -> np.ndarray:
        mask = self.processor.output_prob_to_mask(sparse_prob)
        return mask.cpu().numpy().astype(np.uint8)

    def _restore_locked_classes(self) -> None:
        # the model has no way to "skip" an object it's already tracking (it
        # always predicts every tracked object together), so a locked class
        # would otherwise still silently drift frame to frame. Undo that by
        # discarding the model's fresh guess for each locked class and
        # putting back exactly whatever was already saved to disk for this
        # frame -- not a snapshot from whenever the class was locked, but
        # this specific frame's own previously saved mask (blank if this
        # frame was never annotated before).
        if not self.locked_object_ids:
            return
        saved_mask = self.res_man.get_mask(self.curr_ti)
        for obj_id in self.locked_object_ids:
            self.curr_mask[self.curr_mask == obj_id] = 0
            if self.curr_prob is not None:
                self.curr_prob[obj_id] = 0
            if saved_mask is None:
                continue
            keep = (saved_mask == obj_id)
            if np.any(keep):
                self.curr_mask[keep] = obj_id
                if self.curr_prob is not None:
                    self.curr_prob[:, keep] = 0
                    self.curr_prob[obj_id, keep] = 1

    def _step_with_current_annotation(self, *, force_permanent: bool = False) -> torch.Tensor:
        # only pass a mask (and register objects) when the current frame
        # actually has annotated pixels; otherwise this is just a "predict
        # from memory" step for whatever is already being tracked. Passing
        # objects=[] with an (all-empty) mask would hit a short-circuit in
        # InferenceCore.step() that returns a background-only result without
        # touching the already-tracked objects, which _scatter_sparse_prob
        # can't reconcile with the currently-tracked object count.
        present_ids = self._present_object_ids()
        if present_ids:
            return self.processor.step(self.curr_image_torch,
                                       self._curr_mask_torch(),
                                       objects=present_ids,
                                       idx_mask=True,
                                       force_permanent=force_permanent)
        return self.processor.step(self.curr_image_torch, force_permanent=force_permanent)

    def compose_current_im(self):
        self.vis_image = get_visualization(self.vis_mode, self.curr_image_np, self.curr_mask,
                                           self.overlay_layer, self.vis_target_objects)

    def update_canvas(self):
        self.gui.set_canvas(self.vis_image)

    def update_current_image_fast(self, invalid_soft_mask: bool = False):
        # fast path, uses gpu. Changes the image in-place to avoid copying
        # thus current_image_torch must be voided afterwards
        # do_no_save_soft_mask is an override to solve #41
        self.vis_image = get_visualization_torch(self.vis_mode, self.curr_image_torch,
                                                 self.curr_prob, self.overlay_layer_torch,
                                                 self.vis_target_objects)
        self.curr_image_torch = None
        self.vis_image = np.ascontiguousarray(self.vis_image)
        save_visualization = self.save_visualization_mode in [
            'Propagation only (higher quality)', 'Always'
        ]
        if save_visualization and not invalid_soft_mask:
            self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
        if self.save_soft_mask and not invalid_soft_mask:
            self.res_man.save_soft_mask(self.curr_ti, self.curr_prob.cpu().numpy())
        self.gui.set_canvas(self.vis_image)

    def show_current_frame(self, fast: bool = False, invalid_soft_mask: bool = False):
        # Re-compute overlay and show the image
        if fast:
            self.update_current_image_fast(invalid_soft_mask)
        else:
            self.compose_current_im()
            if self.save_visualization_mode == 'Always':
                self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
            self.update_canvas()

        self.gui.update_slider(self.curr_ti)
        self.gui.frame_name.setText(self.res_man.names[self.curr_ti] + '.jpg')

    def set_vis_mode(self):
        self.vis_mode = self.gui.combo.currentText()
        self.show_current_frame()

    def save_current_mask(self):
        # save mask to hard disk
        self.res_man.save_mask(self.curr_ti, self.curr_mask)

    def on_slider_update(self):
        # if we are propagating, the on_run function will take care of everything
        # don't do duplicate work here
        self.curr_ti = self.gui.tl_slider.value()
        if not self.propagating:
            # with self.vis_cond:
            #     self.vis_cond.notify()
            if self.curr_frame_dirty:
                self.save_current_mask()
            self.curr_frame_dirty = False

            self.reset_this_interaction()
            self.curr_ti = self.gui.tl_slider.value()
            self.load_current_image_mask()
            self.show_current_frame()

    def on_forward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_fn = self.on_next_frame
            self.gui.forward_propagation_start()
            self.propagate_direction = 'forward'
            self.on_propagate()

    def on_backward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_fn = self.on_prev_frame
            self.gui.backward_propagation_start()
            self.propagate_direction = 'backward'
            self.on_propagate()

    def on_pause(self):
        self.propagating = False
        self.gui.text(f'Propagation stopped at t={self.curr_ti}.')
        self.gui.pause_propagation()

    def on_propagate(self):
        # start to propagate
        start_ti = self.curr_ti
        direction = self.propagate_direction
        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()

            self.gui.text(f'Propagation started at t={self.curr_ti}.')
            self.history.log('propagate_start', frame=start_ti, direction=direction)
            self.processor.clear_sensory_memory()
            sparse_prob = self._step_with_current_annotation()
            self.curr_prob = self._scatter_sparse_prob(sparse_prob)
            self.curr_mask = self._mask_from_sparse_prob(sparse_prob)
            self._restore_locked_classes()
            # clear
            self.interacted_prob = None
            self.reset_this_interaction()
            # override this for #41
            self.show_current_frame(fast=True, invalid_soft_mask=True)

            self.propagating = True
            self.gui.clear_all_mem_button.setEnabled(False)
            self.gui.clear_non_perm_mem_button.setEnabled(False)
            self.gui.tl_slider.setEnabled(False)

            dataset = PropagationReader(self.res_man, self.curr_ti, self.propagate_direction)
            loader = get_data_loader(dataset, self.cfg.num_read_workers)

            # propagate till the end
            for data in loader:
                if not self.propagating:
                    break
                self.curr_image_np, self.curr_image_torch = data
                self.curr_image_torch = self.curr_image_torch.to(self.device, non_blocking=True)
                self.propagate_fn()

                sparse_prob = self.processor.step(self.curr_image_torch)
                self.curr_prob = self._scatter_sparse_prob(sparse_prob)
                self.curr_mask = self._mask_from_sparse_prob(sparse_prob)
                self._restore_locked_classes()

                self.save_current_mask()
                self.show_current_frame(fast=True)

                self.update_memory_gauges()
                self.gui.process_events()

                if self.curr_ti == 0 or self.curr_ti == self.T - 1:
                    break

            self.propagating = False
            self.curr_frame_dirty = False
            self.history.log('propagate_stop',
                              direction=direction,
                              start_frame=start_ti,
                              end_frame=self.curr_ti)
            self.on_pause()
            self.on_slider_update()
            self.gui.process_events()

    def pause_propagation(self):
        self.propagating = False

    def on_commit(self):
        if self.interacted_prob is None:
            # get mask from disk
            self.load_current_image_mask()
        else:
            # get mask from interaction
            self.complete_interaction()
            self.update_interacted_mask()

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()
            self.gui.text(f'Permanent memory saved at {self.curr_ti}.')
            sparse_prob = self._step_with_current_annotation(force_permanent=True)
            self.curr_prob = self._scatter_sparse_prob(sparse_prob)
            self._restore_locked_classes()
            self.update_memory_gauges()
            self.update_gpu_gauges()
            self.history.log('commit', frame=self.curr_ti)

    def on_undo(self):
        if self.propagating:
            return

        if self.in_polygon_mode and self.polygon_points:
            removed = self.polygon_points.pop()
            self.hover_first_point = False
            self.gui.text(f'Removed polygon point: {removed}')
            self.history.log('undo', frame=self.curr_ti, target='polygon_point')
            self.compose_polygon_overlay()
            self.update_canvas()
            return

        if not isinstance(self.interaction, ClickInteraction):
            self.gui.text('Nothing to undo.')
            return

        undo_mask = self.click_ctrl.undo()
        if undo_mask is None:
            # Revert to the pre-interaction state
            self.curr_prob = self.interaction.prev_mask.clone()
            self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
            self.save_current_mask()
            self.interacted_prob = None
            self.interaction = None
            self.history.log('undo', frame=self.curr_ti, target='interaction')
            self.show_current_frame()
            return

        self.interaction.obj_mask = undo_mask.to(self.device, non_blocking=True)
        self.interaction.first_click = False
        self.interacted_prob = self.interaction.predict().to(self.device, non_blocking=True)
        self.update_interacted_mask()
        self.update_gpu_gauges()
        self.history.log('undo', frame=self.curr_ti, target='click')

    def on_play_video_timer(self):
        self.curr_ti += 1
        if self.curr_ti > self.T - 1:
            self.curr_ti = 0
        self.gui.tl_slider.setValue(self.curr_ti)

    def on_export_visualization(self):
        # NOTE: Save visualization at the end of propagation
        image_folder = path.join(self.cfg['workspace'], 'visualization', self.vis_mode)
        save_folder = self.cfg['workspace']
        if path.exists(image_folder):
            # Sorted so frames will be in order
            output_path = path.join(save_folder, f'visualization_{self.vis_mode}.mp4')
            self.gui.text(f'Exporting visualization -- please wait')
            self.gui.process_events()
            convert_frames_to_video(image_folder,
                                    output_path,
                                    fps=self.output_fps,
                                    bitrate=self.output_bitrate,
                                    progress_callback=self.gui.progressbar_update)
            self.gui.text(f'Visualization exported to {output_path}')
            self.gui.progressbar_update(0)
        else:
            self.gui.text(f'No visualization images found in {image_folder}')

    def on_export_binary(self):
        # export masks in binary format for other applications, e.g., ProPainter
        mask_folder = path.join(self.cfg['workspace'], 'masks')
        save_folder = path.join(self.cfg['workspace'], 'binary_masks')
        if path.exists(mask_folder):
            os.makedirs(save_folder, exist_ok=True)
            self.gui.text(f'Exporting binary masks -- please wait')
            self.gui.process_events()
            convert_mask_to_binary(mask_folder,
                                   save_folder,
                                   self.vis_target_objects,
                                   progress_callback=self.gui.progressbar_update)
            self.gui.text(f'Binary masks exported to {save_folder}')
            self.gui.progressbar_update(0)
        else:
            self.gui.text(f'No masks found in {mask_folder}')

    def on_fps_dial_change(self):
        self.output_fps = self.gui.fps_dial.value()

    def on_bitrate_dial_change(self):
        self.output_bitrate = self.gui.bitrate_dial.value()

    def update_interacted_mask(self):
        self.curr_prob = self.interacted_prob
        self.curr_mask = torch_prob_to_numpy_mask(self.interacted_prob)
        self.save_current_mask()
        self.show_current_frame()
        self.curr_frame_dirty = False

    def reset_this_interaction(self):
        self.complete_interaction()
        self.interacted_prob = None
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()

    def on_reset_mask(self):
        # preserve locked classes' pixels -- resetting the frame shouldn't
        # be able to touch a class that's protected from editing
        if self.locked_object_ids:
            keep = np.isin(self.curr_mask, list(self.locked_object_ids))
            self.curr_mask[~keep] = 0
        else:
            self.curr_mask.fill(0)
        if self.curr_prob is not None:
            for obj_id in range(self.num_objects + 1):
                if obj_id not in self.locked_object_ids:
                    self.curr_prob[obj_id] = 0
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.history.log('reset_frame', frame=self.curr_ti)
        self.show_current_frame()

    def on_reset_object(self):
        if self.curr_object in self.locked_object_ids:
            self.gui.text(f'Class {self.curr_object} is locked. Unlock it to reset.')
            return
        self.curr_mask[self.curr_mask == self.curr_object] = 0
        if self.curr_prob is not None:
            self.curr_prob[self.curr_object] = 0
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.history.log('reset_object', frame=self.curr_ti, object_id=self.curr_object)
        self.show_current_frame()

    def complete_interaction(self):
        if self.interaction is not None:
            self.interaction = None

    def on_prev_frame(self, step=1):
        new_ti = max(0, self.curr_ti - step)
        self.gui.tl_slider.setValue(new_ti)

    def on_next_frame(self, step=1):
        new_ti = min(self.curr_ti + step, self.length - 1)
        self.gui.tl_slider.setValue(new_ti)

    def update_gpu_gauges(self):
        if 'cuda' in self.device:
            info = torch.cuda.mem_get_info()
            global_free, global_total = info
            global_free /= (2**30)
            global_total /= (2**30)
            global_used = global_total - global_free

            self.gui.gpu_mem_gauge.setFormat(f'{global_used:.1f} GB / {global_total:.1f} GB')
            self.gui.gpu_mem_gauge.setValue(round(global_used / global_total * 100))

            used_by_torch = torch.cuda.max_memory_allocated() / (2**30)
            self.gui.torch_mem_gauge.setFormat(f'{used_by_torch:.1f} GB / {global_total:.1f} GB')
            self.gui.torch_mem_gauge.setValue(round(used_by_torch / global_total * 100 / 1024))
        elif 'mps' in self.device:
            mem_used = mps.current_allocated_memory() / (2**30)
            self.gui.gpu_mem_gauge.setFormat(f'{mem_used:.1f} GB')
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat('N/A')
            self.gui.torch_mem_gauge.setValue(0)
        else:
            self.gui.gpu_mem_gauge.setFormat('N/A')
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat('N/A')
            self.gui.torch_mem_gauge.setValue(0)

    def on_gpu_timer(self):
        self.update_gpu_gauges()

    def update_memory_gauges(self):
        try:
            curr_perm_tokens = self.processor.memory.work_mem.perm_size(0)
            self.gui.perm_mem_gauge.setFormat(f'{curr_perm_tokens} / {curr_perm_tokens}')
            self.gui.perm_mem_gauge.setValue(100)

            max_work_tokens = self.processor.memory.max_work_tokens
            max_long_tokens = self.processor.memory.max_long_tokens

            curr_work_tokens = self.processor.memory.work_mem.non_perm_size(0)
            curr_long_tokens = self.processor.memory.long_mem.non_perm_size(0)

            self.gui.work_mem_gauge.setFormat(f'{curr_work_tokens} / {max_work_tokens}')
            self.gui.work_mem_gauge.setValue(round(curr_work_tokens / max_work_tokens * 100))

            self.gui.long_mem_gauge.setFormat(f'{curr_long_tokens} / {max_long_tokens}')
            self.gui.long_mem_gauge.setValue(round(curr_long_tokens / max_long_tokens * 100))

        except AttributeError as e:
            self.gui.work_mem_gauge.setFormat('Unknown')
            self.gui.long_mem_gauge.setFormat('Unknown')
            self.gui.work_mem_gauge.setValue(0)
            self.gui.long_mem_gauge.setValue(0)

    def on_work_min_change(self):
        if self.initialized:
            self.gui.work_mem_min.setValue(
                min(self.gui.work_mem_min.value(),
                    self.gui.work_mem_max.value() - 1))
            self.update_config()

    def on_work_max_change(self):
        if self.initialized:
            self.gui.work_mem_max.setValue(
                max(self.gui.work_mem_max.value(),
                    self.gui.work_mem_min.value() + 1))
            self.update_config()

    def update_config(self):
        if self.initialized:
            with open_dict(self.cfg):
                self.cfg.long_term['min_mem_frames'] = self.gui.work_mem_min.value()
                self.cfg.long_term['max_mem_frames'] = self.gui.work_mem_max.value()
                self.cfg.long_term['max_num_tokens'] = self.gui.long_mem_max.value()
                self.cfg['mem_every'] = self.gui.mem_every_box.value()

            self.processor.update_config(self.cfg)

    def on_clear_memory(self):
        self.processor.clear_memory()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device:
            mps.empty_cache()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_clear_non_permanent_memory(self):
        self.processor.clear_non_permanent_memory()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device:
            mps.empty_cache()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_import_mask(self):
        file_name = self.gui.open_file('Mask')
        if len(file_name) == 0:
            return

        mask = self.res_man.import_mask(file_name, size=(self.h, self.w))

        shape_condition = ((len(mask.shape) == 2) and (mask.shape[-1] == self.w)
                           and (mask.shape[-2] == self.h))

        object_condition = (mask.max() <= self.num_objects)

        if not shape_condition:
            self.gui.text(f'Expected ({self.h}, {self.w}). Got {mask.shape} instead.')
        elif not object_condition:
            self.gui.text(f'Expected {self.num_objects} objects. Got {mask.max()} objects instead.')
        else:
            self.gui.text(f'Mask file {file_name} loaded.')
            self.curr_image_torch = self.curr_prob = None
            self.curr_mask = mask
            self.show_current_frame()
            self.save_current_mask()
            self.history.log('import_mask', frame=self.curr_ti, file=file_name)

    def on_import_layer(self):
        file_name = self.gui.open_file('Layer')
        if len(file_name) == 0:
            return

        self._try_load_layer(file_name)

    def _try_load_layer(self, file_name):
        try:
            layer = self.res_man.import_layer(file_name, size=(self.h, self.w))

            self.gui.text(f'Layer file {file_name} loaded.')
            self.overlay_layer = layer
            self.overlay_layer_torch = torch.from_numpy(layer).float().to(self.device) / 255
            self.show_current_frame()
        except FileNotFoundError:
            self.gui.text(f'{file_name} not found.')

    def on_save_soft_mask_toggle(self):
        self.save_soft_mask = self.gui.save_soft_mask_checkbox.isChecked()

    def on_mouse_motion_xy(self, x: int, y: int):
        self.last_ex, self.last_ey = x, y

        # Check if polygon is being drawn and at least one point exists
        if self.polygon_points:
            # Check distance to first point
            first_pt = self.polygon_points[0]
            dist = ((x - first_pt[0])**2 + (y - first_pt[1])**2)**0.5
            was_hovering = self.hover_first_point
            self.hover_first_point = dist <= self.hover_threshold

            # If hover state changed, update the canvas
            if self.hover_first_point != was_hovering:
                self.compose_polygon_overlay()
                self.update_canvas()

    def on_toggle_vis_mode(self):
        vis_modes = ['image', 'mask', 'overlay']
        try:
            next_index = (vis_modes.index(self.vis_mode) + 1) % len(vis_modes)
            self.vis_mode = vis_modes[next_index]
        except ValueError:
            self.vis_mode = 'overlay'
        print(f'Visualization mode changed to {self.vis_mode}')

        # Update the dropdown menu to show the current mode
        self.gui.combo.setCurrentText(self.vis_mode)
        self.show_current_frame()

    @property
    def h(self) -> int:
        return self.res_man.h

    @property
    def w(self) -> int:
        return self.res_man.w

    @property
    def T(self) -> int:
        return self.res_man.T
