import torch
import cv2
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import numpy as np

from typing import List, Tuple
from logging import Logger
from omegaconf import DictConfig
from pathlib import Path
from objects.object import Object, Features, ObjectDetector
from blender_saver import blender_save
from optimizer import Optimizer
from initializer import Initializer
from utils_ema.plot import plotter
from utils_ema.image import Image
import matplotlib.pyplot as plt


class Solver:
    def __init__(
        self, cfg: DictConfig, obj: Object, detector: ObjectDetector, logger: Logger
    ) -> None:
        self.logger: Logger = logger
        self.cfg: DictConfig = cfg
        self.initializer = Initializer(cfg, logger, detector, obj)

    def calibrate(self) -> None:

        # initialize scene
        self.logger.info("Initializing scene")
        self.scene = self.initializer.initialize()

        # global calibration
        self.logger.info("Calibrating...")
        optimizer = Optimizer(cfg=self.cfg.calibration.calib_params, scene=self.scene, logger=self.logger, 
                              n_features_min=self.scene.n_features_min)
        optimizer.run()

        # for show purposes
        if self.cfg.calibration.test.calib_show_realtime:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # def plot_colormaps(self, cmap='viridis', point_size=10):
    #
    #     for cam_id in range(self.scene.n_cameras):
    #         x = torch.cat([ self.scene.get_xy(time_id)[0][cam_id] for time_id in range(self.scene.time_instants) ], dim=0).cpu()
    #         y = torch.cat([ self.scene.get_xy(time_id)[1][cam_id] for time_id in range(self.scene.time_instants) ], dim=0).cpu()
    #         # compute distance between x and y
    #         distances = torch.norm(x-y, dim=1).unsqueeze(1)
    #         coords = x
    #
    #         if coords.shape[1] != 2 or distances.shape[1] != 1 or coords.shape[0] != distances.shape[0]:
    #             raise ValueError("coords should have shape [N,2] and distances should have shape [N,1] with matching N.")
    #
    #         # Normalize distances
    #         B_max = distances.max()
    #         B_normalized = distances / B_max if B_max > 0 else distances  # Avoid division by zero
    #
    #         # Convert tensors to NumPy for plotting
    #         A_np = coords.detach().cpu().numpy()
    #         B_np = B_normalized.detach().cpu().numpy().flatten()
    #
    #         # Create plot
    #         fig, ax = plt.subplots(figsize=(6, 5))
    #         sc = ax.scatter(A_np[:, 1], A_np[:, 0], c=B_np, cmap=cmap, s=point_size)
    #
    #         # Add colorbar
    #         cbar = plt.colorbar(sc, ax=ax)
    #         cbar.set_label(f"Distance (Max: {B_max.item():.2f} pixels)")
    #
    #         import ipdb; ipdb.set_trace()
    #         # Show plot
    #         plt.show()

    def plot_colormaps(self, cmap='viridis', point_size=10):
        for cam_id in range(self.scene.n_cameras):

            import ipdb; ipdb.set_trace()
            res = self.scene.cameras[cam_id].intr.resolution
            x_min = 0; y_min = 0; x_max = res[0]; y_max = res[1]

            x = torch.cat([self.scene.get_xy(time_id)[0][cam_id] for time_id in range(self.scene.time_instants)], dim=0).cpu()
            y = torch.cat([self.scene.get_xy(time_id)[1][cam_id] for time_id in range(self.scene.time_instants)], dim=0).cpu()
            
            # Compute distance between x and y
            distances = torch.norm(x - y, dim=1).unsqueeze(1)
            coords = x

            if coords.shape[1] != 2 or distances.shape[1] != 1 or coords.shape[0] != distances.shape[0]:
                raise ValueError("coords should have shape [N,2] and distances should have shape [N,1] with matching N.")

            # Normalize distances
            B_max = distances.max()
            B_normalized = distances / B_max if B_max > 0 else distances  # Avoid division by zero

            # Convert tensors to NumPy for plotting
            A_np = coords.detach().cpu().numpy()
            B_np = B_normalized.detach().cpu().numpy().flatten()

            # Create scatter plot with Plotly
            fig = go.Figure(data=go.Scatter(
                x=A_np[:, 0], y=A_np[:, 1], mode='markers',
                marker=dict(size=point_size, color=B_np, colorscale=cmap, showscale=True,
                            colorbar=dict(title=f"Distance (Max: {B_max.item():.2f} pixels)"))
            ))

            # Flip Y-axis (higher values at the top)
            fig.update_layout(
                title="Scatter Plot with Colormap",
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
                # xaxis=dict(autorange='reversed'),  # Flip the vertical axis
                yaxis=dict(autorange='reversed'),  # Flip the vertical axis
                yaxis_scaleanchor="x",  # Equal aspect ratio (same scale on both axes)
                template="plotly_white"
            )

            fig.add_shape(
                type="rect",
                x0=x_min, y0=y_min,
                x1=x_max, y1=y_max,
                line=dict(color="black", width=2),
            )


            # Show the figure
            fig.show()

    def save(self) -> None:
        # Save calibration results on blender
        blender_save(self.cfg.paths.calib_results_dir, self.scene, self.logger)
 
    def load(self) -> bool:
        return True
