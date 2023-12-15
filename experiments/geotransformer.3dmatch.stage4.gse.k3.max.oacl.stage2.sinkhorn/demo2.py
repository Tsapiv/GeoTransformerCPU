import argparse
import time

import numpy as np
import open3d as o3d
import torch

from config import make_cfg
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.torch import to_cuda, release_cuda
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", required=True, help="src point cloud ply file")
    parser.add_argument("--ref_file", required=True, help="src point cloud ply file")
    parser.add_argument("--voxel_size", type=float, default=None, help="down sample voxel size")
    parser.add_argument("--weights", required=True, help="model weights file")
    return parser


def load_data(args, scale):
    src_pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(args.src_file)
    ref_pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(args.ref_file)

    if args.voxel_size:
        src_pcd_down = src_pcd.voxel_down_sample(args.voxel_size)
        ref_pcd_down = ref_pcd.voxel_down_sample(args.voxel_size)
    else:
        src_pcd_down = src_pcd
        ref_pcd_down = ref_pcd


    src_points = np.asarray(src_pcd_down.points) / scale
    ref_points = np.asarray(ref_pcd_down.points) / scale
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        "transform": np.eye(4).astype(np.float32)
    }

    return data_dict, src_pcd, ref_pcd


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()

    scale = 100

    # cfg.backbone.init_voxel_size = args.voxel_size * 5
    #
    # cfg.backbone.init_radius = cfg.backbone.base_radius * cfg.backbone.init_voxel_size
    # cfg.backbone.init_sigma = cfg.backbone.base_sigma * cfg.backbone.init_voxel_size
    # cfg.model.ground_truth_matching_radius = args.voxel_size

    # prepare data
    data_dict, src_pcd, ref_pcd = load_data(args, scale)
    neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch
    t = time.perf_counter()
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )

    # prepare model
    model = create_model(cfg)
    state_dict = torch.load(args.weights, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict["model"])

    model.eval()

    # prediction
    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    output_dict = release_cuda(output_dict)

    # get results
    estimated_transform = output_dict["estimated_transform"]
    print(time.perf_counter() - t)
    print(estimated_transform)


    # visualization
    # ref_pcd = make_open3d_point_cloud(ref_points)
    # ref_pcd.estimate_normals()
    # ref_pcd.paint_uniform_color(get_color("custom_yellow"))
    # src_pcd = make_open3d_point_cloud(src_points)
    # src_pcd.estimate_normals()
    # src_pcd.paint_uniform_color(get_color("custom_blue"))
    # draw_geometries(ref_pcd, src_pcd)
    estimated_transform[:3, -1] *= scale
    src_pcd = src_pcd.transform(estimated_transform)
    draw_geometries(ref_pcd, src_pcd)



if __name__ == "__main__":
    main()
