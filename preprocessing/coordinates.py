"""Planar frame: +x ego forward, +y ego left, radians for heading."""
import numpy as np
from pyquaternion import Quaternion


def quaternion_yaw(rotation):
    forward = Quaternion(rotation).rotation_matrix[:, 0]
    return float(np.arctan2(forward[1], forward[0]))


def rotation_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def global_to_ego(points, origin, ego_yaw):
    points = np.asarray(points, dtype=np.float64)
    return (points - np.asarray(origin, dtype=np.float64)[:2]) @ rotation_matrix(ego_yaw)


def ego_to_global(points, origin, ego_yaw):
    return np.asarray(points, dtype=np.float64) @ rotation_matrix(ego_yaw).T + np.asarray(origin)[:2]


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def global_heading_to_ego(heading, ego_yaw):
    return wrap_angle(np.asarray(heading) - ego_yaw)


def ego_heading_to_global(heading, ego_yaw):
    return wrap_angle(np.asarray(heading) + ego_yaw)
