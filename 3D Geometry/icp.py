import trimesh
import numpy as np
from sklearn.neighbors import KDTree


# Point to point ICP- https://en.wikipedia.org/wiki/Iterative_closest_point

def icp(surface_points_1, surface_points_2, max_iterations=30, min_dist=2e-2):
    tree_1 = KDTree(surface_points_1)
    R = None
    t = None
    error = []
    for i in range(max_iterations):
        distances, indices = tree_1.query(surface_points_2, 1)
        error.append(distances.mean())
        matching_indices = distances < min_dist
        p = surface_points_1[indices[matching_indices]].squeeze()
        q = np.expand_dims(surface_points_2, axis=1)[
            matching_indices].squeeze()
        p_mean = np.mean(p, axis=0)
        q_mean = np.mean(q, axis=0)
        p_bar = p - p_mean
        q_bar = q - q_mean
        u, sigma, v_t = np.linalg.svd(np.dot(q_bar.T, p_bar))
        det = np.linalg.det(np.dot(v_t.T, u.T))
        correction = np.eye(3)
        if det < 0:
            correction[2, 2] = -1
        R = np.dot(np.dot(v_t.T, correction), u.T)
        t = p_mean.T - np.dot(R, q_mean.T)
        t = np.expand_dims(t, axis=1)
        surface_points_2 = (np.dot(R, surface_points_2.T) + t).T
    distances, indices = tree_1.query(surface_points_2, 1)
    error.append(np.mean(distances))
    return surface_points_2, error


if __name__ == '__main__':
    mm_1 = trimesh.load('mesh1.obj')
    mm_2 = trimesh.load('mesh2.obj')
    m2_t, error = icp(mm_1.vertcies, mm_2.vertcies, 30, 2e-2)
