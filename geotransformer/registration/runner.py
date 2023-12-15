from typing import Optional

import numpy as np
import open3d as o3d

from geotransformer.registration.config import make_cfg
from geotransformer.registration.model import create_model, GeoTransformer
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_tensor, release_tensor


def register_point_clouds(src: o3d.geometry.PointCloud,
                          ref: o3d.geometry.PointCloud,
                          voxel_size: Optional[float] = None,
                          model: Optional[GeoTransformer] = None,
                          scale: Optional[float] = 100):
    cfg = make_cfg()

    if voxel_size:
        src = src.voxel_down_sample(voxel_size)
        ref = ref.voxel_down_sample(voxel_size)

    src_points = np.asarray(src.points) / scale
    ref_points = np.asarray(ref.points) / scale
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        "transform": np.eye(4).astype(np.float32)
    }

    data_dict = registration_collate_fn_stack_mode(
        [data_dict],
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits=[38, 36, 36, 38]
    )

    # prepare model
    if model is None:
        model = create_model(cfg)

    # prediction
    data_dict = to_tensor(data_dict)
    output_dict = model(data_dict)
    output_dict = release_tensor(output_dict)

    # get results
    estimated_transform = output_dict["estimated_transform"]
    estimated_transform[:3, -1] *= scale

    return estimated_transform
