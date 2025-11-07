import numpy as np

# dewpoint formula (Magnus / Alduchov–Eskridge constants)
A = 17.62
B = 243.12  # deg C


def td_from_t_rh(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    gamma = np.log(np.clip(rh, 1e-6, 100.0) / 100.0) + (A * t_c) / (B + t_c)
    td = (B * gamma) / (A - gamma)
    return td


def met_dir_to_uv(
    speed: np.ndarray, direction_deg_from_north: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    phi = np.deg2rad(direction_deg_from_north)  # from north, clockwise
    # components (m/s if speed in m/s)
    u = -speed * np.sin(phi)
    v = -speed * np.cos(phi)
    return u, v
