import torch
import plotly.graph_objects as go
from blender_saver import blender_save
from logging import Logger
from omegaconf import DictConfig
from objects.object import Object, ObjectDetector
from optimizer import Optimizer
from initializer import Initializer
from pathlib import Path


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


    def save_colormaps(self, cmap='viridis', point_size=10):

        x_tens, y_tens, mask = self.scene.get_xy(pixel_unit=True)
        dir = Path(self.cfg.paths.calib_results_dir) / "colormaps"
        if not dir.exists():
            dir.mkdir(parents=True)

        for cam_id in range(self.scene.n_cameras):

            idxs = mask[:,cam_id,...].reshape(-1).detach().cpu()
            x = x_tens[:,cam_id,...].reshape(-1,2)[idxs].detach().cpu()
            y = y_tens[:,cam_id,...].reshape(-1,2)[idxs].detach().cpu()
            distances = torch.norm(x - y, dim=1).unsqueeze(1).detach().cpu()
            coords = x

            res = self.scene.cameras.intr.resolution[0,cam_id,0,...].type(torch.int64).detach().cpu()
            x_min = 0; y_min = 0; x_max = res[0]; y_max = res[1]

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

            # save figure
            filename = f"colormap_cam_{cam_id}.html"
            path = dir / filename
            fig.write_html(path)

    def save(self) -> None:

        # scene postprocess and save
        self.scene.scene_postprocess_and_save_data()

        # Save calibration results on blender
        blender_save(self.cfg.paths.calib_results_dir, self.scene, self.logger)

        # Save camera data
        self.save_colormaps()
 
    def load(self) -> bool:
        return True
