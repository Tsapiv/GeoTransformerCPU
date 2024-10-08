import argparse

import open3d as o3d
from geotransformer.registration.runner import register_point_clouds


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", required=True, help="src point cloud ply file")
    parser.add_argument("--ref_file", required=True, help="src point cloud ply file")
    parser.add_argument("--voxel_size", type=float, default=1, help="down sample voxel size")
    parser.add_argument("--scale", type=float, default=None, help="point cloud scaling factor")

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    src_pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(args.src_file)
    ref_pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(args.ref_file)

    estimated_transform = register_point_clouds(src_pcd, ref_pcd, voxel_size=args.voxel_size, scale=args.scale)

    src_pcd = src_pcd.transform(estimated_transform)
    o3d.visualization.draw_geometries([ref_pcd, src_pcd])


if __name__ == "__main__":
    main()
