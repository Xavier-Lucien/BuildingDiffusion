import os

import numpy as np
import trimesh


def _load_as_mesh(model_path):
    """Load path as a single trimesh.Trimesh object."""
    loaded = trimesh.load(model_path)
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError(f"Model has no geometry: {model_path}")
        return loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported model type: {type(loaded).__name__}")
    return loaded


def voxelize_model(model_path, voxel_resolution=128, save_path=None, return_metadata=False):
    """Voxelize a 3D model to a fixed occupancy grid.

    Args:
        model_path (str): Path to the mesh file.
        voxel_resolution (int): Target grid size per axis, e.g. 128 -> 128^3.
        save_path (str | None): Optional output path (.npy or .npz).
        return_metadata (bool): If True, also return normalization metadata.

    Returns:
        np.ndarray[bool] or (np.ndarray[bool], dict): Occupancy grid with shape
        (voxel_resolution, voxel_resolution, voxel_resolution). If
        return_metadata=True, also returns transform metadata.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if int(voxel_resolution) <= 0:
        raise ValueError("voxel_resolution must be a positive integer")

    voxel_resolution = int(voxel_resolution)
    mesh = _load_as_mesh(model_path)

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {model_path}")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Mesh contains non-finite vertices")

    bounds = mesh.bounds
    min_bound = bounds[0].astype(np.float64)
    max_bound = bounds[1].astype(np.float64)
    extents = max_bound - min_bound
    longest = float(np.max(extents))
    if longest <= 0.0:
        raise ValueError("Mesh has zero size bounding box")

    # Use a pitch derived from longest bbox edge so the largest dimension
    # maps to the requested grid resolution.
    pitch = longest / voxel_resolution
    voxel_grid = mesh.voxelized(pitch=pitch)

    occupied = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution), dtype=bool)
    if voxel_grid.points is not None and len(voxel_grid.points) > 0:
        normalized = (voxel_grid.points - min_bound) / longest
        indices = np.floor(normalized * (voxel_resolution - 1)).astype(np.int64)
        indices = np.clip(indices, 0, voxel_resolution - 1)
        occupied[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    metadata = {
        "min_bound": min_bound,
        "max_bound": max_bound,
        "longest_extent": longest,
        "pitch": pitch,
        "voxel_resolution": voxel_resolution,
    }

    if save_path is not None:
        save_ext = os.path.splitext(save_path)[1].lower()
        if save_ext == ".npz":
            np.savez_compressed(save_path, voxels=occupied.astype(np.uint8), **metadata)
        elif save_ext in ("", ".npy"):
            np.save(save_path, occupied)
        else:
            raise ValueError("save_path must end with .npy or .npz")

    if return_metadata:
        return occupied, metadata
    return occupied
