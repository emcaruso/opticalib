from omegaconf import DictConfig
import cv2
from logging import Logger
from scene import Scene
from objects.object import Object, Features, ObjectDetector
from typing import List
from sensorflow.src.collector import CollectorLoader
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from optimizer import Optimizer
from utils_ema.camera_cv import Camera_cv, Intrinsics
from utils_ema.geometry_pose import Pose
from utils_ema.geometry_euler import eul
from utils_ema.plot import plotter


class Initializer():
    def __init__(self, cfg: DictConfig, logger: Logger, detector: ObjectDetector,  obj: Object):
        self.cfg = cfg
        self.logger = logger
        self.detector = detector
        self.obj = obj

    def initialize(self):

        # get scene
        scene = self.__get_scene()

        # init board relative poses + intrinsics (fixed focal length)
        self.__precalibration(scene)
        self.logger.info("Pre-calibrated intrinsics (fixed focal length), and relative poses")

        # init cameras extrinsics
        self.__init_cameras_extrinsics(scene)
        self.logger.info("Initialize camera extrinsics")

        # optimize to get object poses
        self.logger.info("Pre-calibrate object poses")
        optimizer = Optimizer(cfg=self.cfg.calibration.precalib_params, scene=scene, logger=self.logger,
                            intr_K=False, intr_D=False, extr=False, obj_rel=False, obj_pose=True,
                            n_features_min=scene.n_features_min,
                             )
        optimizer.run()
        
        
        return scene

    def __get_scene(self) -> Scene:
        features_gt = self.__collect_features_gt()
        cameras = self.__get_cameras(features_gt)
        objects = self.__get_objects(features_gt)
        scene = Scene(self.cfg, self.logger, features_gt, cameras, objects)
        return scene

    def __precalibration(self, scene: Scene):
        '''
        get object relative poses + camera intrinsics
        '''

        relative_poses = {}

        points_3D, points_2D, infos = scene.get_3D_2D_points_foreach_cam(filter_boards_by_n_features_min = True, numpy=True)

        for cam_id, p3D, p2D, info in zip(range(len(points_3D)), points_3D, points_2D, infos):
            
            # init relative poses to empty list
            for obj in scene.objects:
                obj.relative_poses = [ None for _ in range(obj.params.n_boards)]

            # if cam_id == 1: continue
            cam = scene.cameras[cam_id]
            K = cam.intr.K_pix.cpu().numpy()
            D = cam.intr.D_params.cpu().numpy()
            flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_FOCAL_LENGTH
            # flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT
            # flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |  
            #         cv2.CALIB_FIX_TANGENT_DIST | cv2.CALIB_FIX_FOCAL_LENGTH)
            # flags = cv2.CALIB_USE_INTRINSIC_GUESS
            imageSize = tuple(cam.intr.resolution.numpy())
            # res = cv2.calibrateCamera( objectPoints=p3D, imagePoints=p2D, imageSize=imageSize, cameraMatrix=None, distCoeffs=None)
            res = cv2.calibrateCamera( objectPoints=p3D, imagePoints=p2D, imageSize=imageSize, cameraMatrix=K, distCoeffs=D, flags=flags)
            self.logger.info(f"Precalib error for camera {cam_id}: {res[0]}")

            K_params = torch.FloatTensor([res[1][0,0], res[1][1,1], res[1][0,2], res[1][1,2]])
            K_params *= cam.intr.unit_pixel_ratio()
            D_params = torch.tensor(res[2]).squeeze()
            cam.intr.K_params = K_params
            cam.intr.D_params = D_params
            cam.intr.update_intrinsics()

            for i, info in enumerate(infos[cam_id]):

                pose = Pose.cvvecs2pose(res[3][i], res[4][i] * cam.intr.unit_pixel_ratio().item())
                time_id = info["time_id"]
                board_id = info["board_id"]
                scene.objects[time_id].relative_poses[board_id] = pose

            for time_id in range(scene.time_instants):

                # reference pose
                obj = scene.objects[time_id]
                pose_ref = obj.relative_poses[0]
                if pose_ref is None:
                    continue

                for board_id in range(1,scene.objects[0].params.n_boards):

                    # relative pose
                    pose_rel = obj.relative_poses[board_id]
                    if pose_rel is None:
                        continue

                    if board_id not in relative_poses.keys():
                        relative_poses[board_id] = []

                    relative_pose = pose_ref - pose_rel
                    relative_poses[board_id] += [relative_pose]


        # reinit objects with shared relative pose
        scene.objects = self.__get_objects(scene.features_gt, same_relative_poses=True)
        for board_id in relative_poses.keys():
            r = relative_poses[board_id]
            average_pose = Pose.average_poses(r)

            for obj in scene.objects:
                obj.relative_poses[board_id] = average_pose


        
    def __init_cameras_extrinsics(self, scene: Scene):
        
        # get first valid time_id
        best_valid_boards = None
        best_time_id = None
        best_score = 0
        for time_id in range(scene.time_instants):
            valid_cams = True
            valid_boards = []
            for cam_id in range(scene.n_cameras):
                valid_boards.append([])
                features = scene.features_gt[time_id][cam_id]
                # if at least one board has enough points in the image, camera can be properly estimated
                for board_id in range(scene.objects[0].params.n_boards):
                    if len(features.points[board_id]) >= scene.n_features_min:
                        valid_boards[cam_id].append(board_id)
                if valid_boards[cam_id] == []:
                    valid_cams = False
                    break

            if not valid_cams:
                continue

            score = sum([ len(valid_boards[cam_id]) for cam_id in range(scene.n_cameras)])
            if score > best_score:
                best_score = score
                best_time_id = time_id
                best_valid_boards = valid_boards

        if best_time_id is None:
            raise Exception("No valid time_id found to initialize cameras extrinsics")

        # init cameras
        # print(best_time_id, best_score)
        self.logger.info(f"Best time_id found to initialize cameras extrinsics: {best_time_id}, n of valid boards: {best_score}")
        points = scene.objects[best_time_id].points()
        ids = scene.objects[best_time_id].ids()

        for cam_id, valid_boards_list in enumerate(best_valid_boards):
            cam = scene.cameras[cam_id]
            ratio = cam.intr.pixel_unit_ratio().item()

            p2D = []
            p3D = []

            for board_id in valid_boards_list:
                ids2D = scene.features_gt[best_time_id][cam_id].ids[board_id]
                p3D_ = points[board_id].cpu().numpy() * ratio
                ids3D = ids[board_id]
                idxs = torch.where(torch.isin(ids3D, ids2D))[0]
                p3D.append(p3D_[idxs])
                p2D.append(scene.features_gt[best_time_id][cam_id].points[board_id].cpu().numpy())
                res, rvec, tvec = cv2.solvePnP(p3D[-1], p2D[-1], cam.intr.K_pix.cpu().numpy(), cam.intr.D_params.cpu().numpy())

            p2D = np.concatenate(p2D, axis=0)
            p3D = np.concatenate(p3D, axis=0)

            res, rvec, tvec = cv2.solvePnP(p3D, p2D, cam.intr.K_pix.cpu().numpy(), cam.intr.D_params.cpu().numpy())
            _, rvec_list, tvec_list, repr_list = cv2.solvePnPGeneric(p3D, p2D, cam.intr.K_pix.cpu().numpy(), cam.intr.D_params.cpu().numpy())
            idx_best = [np.array_equal(a, tvec) for a in tvec_list].index(True)

            rvec = rvec_list[idx_best]
            tvec = tvec_list[idx_best]
            repr = repr_list[idx_best].item()
            self.logger.info(f"Reprojection error during extrinsics initialization for camera {cam_id}: {repr}")
            if res is False:
                raise Exception("solvePnP failed in init camera extrinsics")
            tvec *= 1/ratio

            cam_pose = Pose.cvvecs2pose(rvec, tvec)
            cam_pose.invert()
            scene.cameras[cam_id].pose = cam_pose

            # # reproject p3D on camera and check the distance with p2D
            # pa = torch.tensor(p3D * (1/ratio))
            # p2D_hat = cam.project_points(pa, longtens=False, und=False)
            # dist = torch.norm(torch.tensor(p2D).squeeze(1) - p2D_hat, dim=1).mean()
            #
            #
            # cam_poses.append(cam_pose)

            # for p in cam_poses:
            #     print(p.position)
            
            # cam_pose = Pose.average_poses(cam_poses)

        # plot for test purposes
        if self.cfg.calibration.test.init_show:
            for cam in scene.cameras:
                plotter.plot_pose(cam.pose)
            for p in points:
                plotter.plot_points(p)
            plotter.show()


        

    def __get_cameras(self, features_gt) -> List[Camera_cv]:
        cameras = []
        n_cameras = len(features_gt[0])
        for i in range(n_cameras):

            # get resolution
            resolution = CollectorLoader.resolutions[i]
            info = CollectorLoader.load_info(self.cfg.paths.collection_dir)
            pixel_size = torch.FloatTensor(info[f"cam_{i:03d}"].PixelSizeMicrometers)*1e-6
            sensor_size = resolution*pixel_size
            crop_offset = torch.FloatTensor(info[f"cam_{i:03d}"]["crop_offset"])
            resolution_native = torch.FloatTensor(info[f"cam_{i:03d}"]["resolution_native"])

            # prior intrinsics
            f = self.cfg.calibration.focal_length_prior
            c = (resolution_native/2 - crop_offset)*pixel_size
            K=torch.FloatTensor([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]])
            D = torch.zeros(5)

            intr = Intrinsics(K=K, D=D, resolution=resolution, sensor_size=sensor_size)
            pose = Pose(position=torch.tensor([0,0,0], dtype=torch.float32), euler=eul(torch.zeros(3, dtype=torch.float32)))
            cameras.append(Camera_cv(device=self.cfg.calibration.device, intrinsics=intr, pose=pose))
        return cameras

    def __get_objects(self, features_gt, same_relative_poses = True) -> List[Object]:
        time_instants = len(features_gt)
        objects = [
            self.obj.clone(same_pose=False, same_relative_poses=same_relative_poses)
            for _ in range(time_instants)
        ]
        return objects

    def __collect_features_gt(self) -> List[List[Features]]:
        self.coll_data = CollectorLoader.load_images(self.cfg.paths.collection_dir, in_ram=self.cfg.collect.in_ram, raw=True)
        next(self.coll_data) # initialize static members
        if not Path(self.cfg.paths.features_file).exists():
            features = []
            self.logger.info("Collecting features...")
            progress_bar = tqdm(self.coll_data, desc="Collecting features", total = CollectorLoader.n_images)
            for images in progress_bar:
                features.append(self.detector.detect_features(images, device=self.cfg.calibration.device))
            # save custom object with pickle
            np.save(self.cfg.paths.features_file, features, allow_pickle=True)


        else:
            features = np.load(self.cfg.paths.features_file, allow_pickle=True)
            for feat in features:
                for f in feat:
                    f.to(self.cfg.calibration.device)
        return features
