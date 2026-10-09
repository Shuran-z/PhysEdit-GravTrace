"""Analytic centre trajectories. Time tau is measured from the declared initial state."""
from __future__ import annotations

import numpy as np


def ballistic(tau: np.ndarray, x0, v0, g: float, g_dir, drag: float = 0.0) -> np.ndarray:
    """Free flight: x0 + v0 tau + g g_dir tau^2 / 2, or with a declared linear drag dv/dt = g g_dir - drag v (1/s)."""
    if drag * float(np.max(tau, initial=0.0)) < 1e-6:
        return x0 + np.outer(tau, v0) + 0.5 * g * np.outer(tau * tau, g_dir)
    v_end = g * np.asarray(g_dir, dtype=float) / drag  # terminal velocity
    return x0 + np.outer(tau, v_end) + np.outer(-np.expm1(-drag * tau) / drag, np.asarray(v0, dtype=float) - v_end)


def torque_free(tau: np.ndarray, rotation0: np.ndarray, omega_world, inertia, step: float = 1.0 / 960.0,
                damping: float = 0.0) -> np.ndarray:
    """Orientation (len(tau), 3, 3) of a free rigid body: Euler's equations in the body frame with principal
    moments `inertia`, integrated by RK4 from `rotation0` and the world angular velocity at tau = 0. A declared
    angular damping torque -damping * omega needs `inertia` in kg m^2 (otherwise any scale will do).
    Gravity exerts no torque about the centre of mass, so this does not depend on g."""
    inertia = np.asarray(inertia, dtype=float)
    w = rotation0.T @ np.asarray(omega_world, dtype=float)
    q = np.array([0.0, 0.0, 0.0, 1.0])  # body rotation since tau = 0, xyzw

    def deriv(q, w):
        x, y, z, s = q
        wx, wy, wz = w
        dq = 0.5 * np.array([s * wx + y * wz - z * wy, s * wy + z * wx - x * wz, s * wz + x * wy - y * wx, -x * wx - y * wy - z * wz])
        dw = np.array([(inertia[1] - inertia[2]) * w[1] * w[2], (inertia[2] - inertia[0]) * w[2] * w[0],
                       (inertia[0] - inertia[1]) * w[0] * w[1]]) / inertia - damping * w / inertia
        return dq, dw

    out, now = np.empty((len(tau), 3, 3)), 0.0
    for i in np.argsort(tau):
        while now < tau[i] - 1e-12:
            h = min(step, tau[i] - now)
            k1 = deriv(q, w)
            k2 = deriv(q + 0.5 * h * k1[0], w + 0.5 * h * k1[1])
            k3 = deriv(q + 0.5 * h * k2[0], w + 0.5 * h * k2[1])
            k4 = deriv(q + h * k3[0], w + h * k3[1])
            q = q + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            w = w + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            q /= np.linalg.norm(q)
            now += h
        x, y, z, s = q
        body = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * s), 2 * (x * z + y * s)],
                         [2 * (x * y + z * s), 1 - 2 * (x * x + z * z), 2 * (y * z - x * s)],
                         [2 * (x * z - y * s), 2 * (y * z + x * s), 1 - 2 * (x * x + y * y)]])
        out[i] = rotation0 @ body
    return out


def incline(tau: np.ndarray, x0, speed: float, g: float, slope: dict, g_dir, top: float = np.inf) -> np.ndarray:
    """Slide (or roll, for a solid sphere) along the slope; `speed` < 0 starts upslope.

    Moving up, gravity and kinetic friction both decelerate the object. If it reaches the top
    edge, `top` metres upslope, it leaves the ramp in free flight; otherwise it stops and slides
    back down (or stays, when static friction holds it). The floor at the bottom is handled by
    the caller as a contact that ends the fit window.
    """
    d = np.asarray(slope["dir"], dtype=float)
    sin, cos = np.sin(np.radians(slope["angle_deg"])), np.cos(np.radians(slope["angle_deg"]))
    mu = float(slope.get("mu", 0.0))
    if slope.get("rolling"):
        a_up = a_down = 5.0 / 7.0 * g * sin
    else:
        a_up, a_down = g * (sin + mu * cos), g * (sin - mu * cos)
    if speed < 0.0:  # decelerate upslope, then return
        t_top = -speed / max(a_up, 1e-9)
        after = np.maximum(tau - t_top, 0.0)
        s = np.where(tau <= t_top, speed * tau + 0.5 * a_up * tau * tau, 0.5 * speed * t_top + 0.5 * max(a_down, 0.0) * after * after)
    elif a_down >= 0.0:
        s = speed * tau + 0.5 * a_down * tau * tau
    else:  # friction wins: decelerate to rest
        t = np.minimum(tau, speed / -a_down)
        s = speed * t + 0.5 * a_down * t * t
    x = x0 + np.outer(s, d)
    reach = speed * speed - 2.0 * a_up * top
    if speed < 0.0 and reach >= 0.0:  # leaves over the top edge
        t_exit = (-speed - np.sqrt(reach)) / a_up
        flying = tau > t_exit
        x[flying] = ballistic(tau[flying] - t_exit, x0 - top * d, (speed + a_up * t_exit) * d, g, g_dir)
    return x
