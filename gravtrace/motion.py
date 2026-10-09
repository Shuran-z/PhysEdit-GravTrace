"""Analytic centre trajectories. Time tau is measured from the declared initial state."""
from __future__ import annotations

import numpy as np


def ballistic(tau: np.ndarray, x0, v0, g: float, g_dir) -> np.ndarray:
    """Free flight: x0 + v0 tau + g g_dir tau^2 / 2."""
    return x0 + np.outer(tau, v0) + 0.5 * g * np.outer(tau * tau, g_dir)


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
