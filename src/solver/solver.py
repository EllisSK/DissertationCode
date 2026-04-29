# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import numpy as np

from typing import Callable
import matplotlib.pyplot as plt

def apply_bcs(Q: np.ndarray, flow_rate: float, zb: np.ndarray):
    g = 9.81

    Q[0, 0] = Q[1, 0] - zb[1, 0] + zb[0, 0]
    Q[0, 1] = flow_rate

    h_interior = np.maximum(Q[-2, 0] - zb[-2, 0], 1e-6)
    q_out = Q[-2, 1]
    
    h_c = (q_out**2 / g)**(1/3)
    
    if h_interior < h_c:
        Q[-1, 0] = Q[-2, 0] - zb[-2, 0] + zb[-1, 0]
    else:
        Q[-1, 0] = h_c
    Q[-1, 1] = q_out

def get_flux(Q: np.ndarray, zb: np.ndarray):
    eta = Q[:, 0]
    q = Q[:, 1]
    z_flat = zb[:, 0]
    g = 9.81

    h = np.maximum(eta - z_flat, 0.0)
    u = np.divide(q, h, out=np.zeros_like(q), where=h > 0)
    F = np.zeros_like(Q)

    F[:, 0] = q
    F[:, 1] = (q * u) + (0.5 * g * (np.power(eta, 2) - (2 * eta * z_flat)))

    return F

def get_source(Q: np.ndarray, zb: np.ndarray, dx: float, mannings_fn: Callable):
    eta = Q[:, 0]
    q = Q[:, 1]
    g = 9.81

    h = np.maximum(eta - zb[:, 0], 0.0)
    dzb_dx = np.gradient(zb[:, 0], dx)
    bed_slope = -g * eta * dzb_dx

    mannings_n = mannings_fn(h)

    R = np.divide(h, 1 + 2 * h, out=np.zeros_like(h), where=h > 1e-6)
    friction = np.divide(g * (mannings_n**2) * q * np.abs(q), h * (R**(4/3)), out=np.zeros_like(q), where=h > 1e-6)

    S = np.zeros_like(Q)
    S[:, 1] = bed_slope - friction

    return S

def hll_flux(Q_L: np.ndarray, Q_R: np.ndarray, zb_interface: np.ndarray):
    eta_L = Q_L[:, 0]
    q_L = Q_L[:, 1]
    eta_R = Q_R[:, 0]
    q_R = Q_R[:, 1]
    z_int = zb_interface[:, 0]
    g = 9.81

    h_L = np.maximum(eta_L - z_int, 0.0)
    h_R = np.maximum(eta_R - z_int, 0.0)

    u_L = np.divide(q_L, h_L, out=np.zeros_like(q_L), where=h_L > 0.0)
    u_R = np.divide(q_R, h_R, out=np.zeros_like(q_R), where=h_R > 0.0)

    a_L = np.sqrt(g * h_L)
    a_R = np.sqrt(g * h_R)

    S_L = np.minimum(u_L - a_L, u_R - a_R)
    S_R = np.maximum(u_L + a_L, u_R + a_R)

    dry_L = h_L == 0.0
    dry_R = h_R == 0.0

    S_L[dry_L] = u_R[dry_L] - 2 * a_R[dry_L]
    S_R[dry_R] = u_L[dry_R] - 2 * a_L[dry_R]

    F_L = get_flux(Q_L, zb_interface)
    F_R = get_flux(Q_R, zb_interface)

    F_int = np.zeros_like(Q_L)

    cond_L = S_L >= 0
    cond_R = S_R <= 0
    cond_star = ~(cond_L | cond_R)

    F_int[cond_L] = F_L[cond_L]
    F_int[cond_R] = F_R[cond_R]

    SL_star = S_L[cond_star, np.newaxis]
    SR_star = S_R[cond_star, np.newaxis]

    F_int[cond_star] = (
        (SR_star * F_L[cond_star]) - (SL_star * F_R[cond_star]) +
        (SL_star * SR_star * (Q_R[cond_star] - Q_L[cond_star]))
    ) / (SR_star - SL_star)

    return F_int

def spatial_reconstructor(Q: np.ndarray, dx: float):
    grad_U = np.zeros_like(Q)

    diff_bwd = (Q[1:-1] - Q[:-2]) / dx
    diff_fwd = (Q[2:] - Q[1:-1]) / dx
    grad_U[1:-1] = 0.5 * (np.sign(diff_bwd) + np.sign(diff_fwd)) * np.minimum(np.abs(diff_bwd), np.abs(diff_fwd))

    Q_L = Q[:-1] + (0.5 * dx * grad_U[:-1])
    Q_R = Q[1:] - (0.5 * dx * grad_U[1:])

    return Q_L, Q_R

def dynamic_timestep(Q: np.ndarray, zb: np.ndarray, dx: float):
    h = np.maximum(Q[:, 0] - zb[:, 0], 0.0)
    u = np.divide(Q[:, 1], h, out=np.zeros_like(Q[:, 1]), where=h > 0.0)
    g = 9.81

    a_dt = np.sqrt(g * h)
    max_speed = np.max(np.abs(u) + a_dt)

    return 0.49 * np.nextafter(dx / max_speed, -np.inf)

def simulate(flow_rate: float, bed_function: Callable | None, mannings_function: Callable):
    length = 12.5
    resolution = 0.1
    start_depth = 0.05

    N = int(length / resolution)
    dx = resolution

    Q_array = np.zeros((N + 2, 2))
    x_vals = np.linspace(dx / 2, length - (dx / 2), N)

    x_vals_full = np.concatenate(([0.0], x_vals, [length]))

    if bed_function:
        zb = bed_function(x_vals_full).reshape((N + 2, 1))
        zb_interface = 0.5 * (zb[:-1] + zb[1:])
    else:
        zb = np.zeros((N + 2, 1))
        zb_interface = np.zeros((N + 1, 1))

    Q_array[:, 0] = start_depth + zb[:, 0]
    Q_array[:, 1] = flow_rate

    t = 0.0
    max_change = 1.0

    convergence_threshold = 5e-5
    max_time = 600

    apply_bcs(Q_array, flow_rate, zb)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"SWE Solver | Flow: {flow_rate*1000} l/s")

    line_eta, = ax1.plot(x_vals_full, Q_array[:, 0], label="Water Elevation (eta)", color="blue", lw=2)
    line_zb, = ax1.plot(x_vals_full, zb[:, 0], label="Bed Elevation (zb)", color="black", lw=2, linestyle='--')
    ax1.set_ylabel("Elevation (m)")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    line_u, = ax2.plot(x_vals_full, np.zeros_like(x_vals_full), label="Velocity (u)", color="red", lw=2)
    ax2.set_xlabel("Distance along flume (m)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    step = 0
    plot_freq = 100

    while t < max_time and max_change > convergence_threshold:
        Q_n = Q_array.copy()

        dt = dynamic_timestep(Q_array, zb, dx)
        t += dt

        Q_L, Q_R = spatial_reconstructor(Q_array, dx)
        F_int = hll_flux(Q_L, Q_R, zb_interface)
        flux_grad = (F_int[1:] - F_int[:-1]) / dx
        S = get_source(Q_array, zb, dx, mannings_function)

        K1 = np.zeros_like(Q_array)
        K1[1:-1] = -flux_grad + S[1:-1]
        U_star = Q_array + (dt * K1)

        Q_array[:] = U_star
        apply_bcs(Q_array, flow_rate, zb)

        Q_L_star, Q_R_star = spatial_reconstructor(Q_array, dx)
        F_int_star = hll_flux(Q_L_star, Q_R_star, zb_interface)
        flux_grad_star = (F_int_star[1:] - F_int_star[:-1]) / dx
        S_star = get_source(Q_array, zb, dx, mannings_function)

        K2 = np.zeros_like(Q_array)
        K2[1:-1] = -flux_grad_star + S_star[1:-1]

        Q_array[:] = Q_n + 0.5 * dt * (K1 + K2)
        apply_bcs(Q_array, flow_rate, zb)

        max_change = np.max(np.abs(Q_array[:, 0] - Q_n[:, 0])) / dt

        if step % plot_freq == 0:
            eta = Q_array[:, 0]
            h = np.maximum(eta - zb[:, 0], 0.0)
            
            u = np.divide(Q_array[:, 1], h, out=np.zeros_like(h), where=h > 1e-4)

            line_eta.set_ydata(eta)
            line_u.set_ydata(u)

            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            ax1.set_title(f"Time: {t:.2f} s | Max Change: {max_change:.6f}")

            fig.canvas.draw()
            fig.canvas.flush_events()
            
        step += 1

    plt.ioff()
    plt.close()

    return t, Q_array

def simulate_barrier(flow_rate: float, bed_function: Callable | None, mannings_function: Callable, barrier_function: Callable, barrier_label: str | None = None):
    length = 12.5
    resolution = 0.1
    start_depth = 0.05
    barrier_x = 5.0
    g = 9.81

    N = int(length / resolution)
    dx = resolution
    barrier_idx = int(round(barrier_x / dx))

    Q_array = np.zeros((N + 2, 2))
    x_vals = np.linspace(dx / 2, length - (dx / 2), N)
    x_vals_full = np.concatenate(([0.0], x_vals, [length]))

    if bed_function:
        zb = bed_function(x_vals_full).reshape((N + 2, 1))
        zb_interface = 0.5 * (zb[:-1] + zb[1:])
    else:
        zb = np.zeros((N + 2, 1))
        zb_interface = np.zeros((N + 1, 1))

    Q_array[:, 0] = start_depth + zb[:, 0]
    Q_array[:, 1] = 0.0

    t = 0.0
    max_change = 1.0
    convergence_threshold = 1e-4
    max_time = 36000

    apply_bcs(Q_array, flow_rate, zb)

    def barrier_face_fluxes(eta_L, eta_R, z_int):
        h_L = max(float(eta_L - z_int), 0.0)
        h_R = max(float(eta_R - z_int), 0.0)
        q_b, M_jet = barrier_function(max(h_L, 1e-9), h_R)
        q_b = float(q_b)
        M_jet = float(M_jet)
        h_L_safe = max(h_L, 1e-6)
        F_L = np.array([
            q_b,
            q_b * q_b / h_L_safe + 0.5 * g * (eta_L * eta_L - 2.0 * eta_L * z_int),
        ])
        F_R = np.array([
            q_b,
            M_jet + 0.5 * g * (eta_R * eta_R - 2.0 * eta_R * z_int),
        ])
        return F_L, F_R

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    title = f"SWE Solver | Flow: {flow_rate * 1000:.0f} l/s"
    if barrier_label:
        title += f" | Barrier: {barrier_label}"
    fig.suptitle(title)

    line_eta, = ax1.plot(x_vals_full, Q_array[:, 0], label="Water Elevation (eta)", color="blue", lw=2)
    line_zb,  = ax1.plot(x_vals_full, zb[:, 0],     label="Bed Elevation (zb)",   color="black", lw=2, linestyle='--')
    ax1.axvline(barrier_x, color='red', lw=1.5, alpha=0.7, linestyle=':', label='Barrier')
    ax1.set_ylabel("Elevation (m)")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    line_u, = ax2.plot(x_vals_full, np.zeros_like(x_vals_full), label="Velocity (u)", color="red", lw=2)
    ax2.axvline(barrier_x, color='red', lw=1.5, alpha=0.7, linestyle=':')
    ax2.set_xlabel("Distance along flume (m)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    step = 0
    plot_freq = 100

    while max_change > convergence_threshold and t < max_time:
        Q_n = Q_array.copy()

        dt = dynamic_timestep(Q_array, zb, dx)
        t += dt

        Q_L, Q_R = spatial_reconstructor(Q_array, dx)
        F_int = hll_flux(Q_L, Q_R, zb_interface)
        flux_grad = (F_int[1:] - F_int[:-1]) / dx

        F_b_L, F_b_R = barrier_face_fluxes(
            Q_L[barrier_idx, 0], Q_R[barrier_idx, 0], zb_interface[barrier_idx, 0]
        )
        flux_grad[barrier_idx - 1] = (F_b_L - F_int[barrier_idx - 1]) / dx
        flux_grad[barrier_idx]     = (F_int[barrier_idx + 1] - F_b_R) / dx

        S = get_source(Q_array, zb, dx, mannings_function)
        K1 = np.zeros_like(Q_array)
        K1[1:-1] = -flux_grad + S[1:-1]
        U_star = Q_array + (dt * K1)

        Q_array[:] = U_star
        apply_bcs(Q_array, flow_rate, zb)

        Q_L_s, Q_R_s = spatial_reconstructor(Q_array, dx)
        F_int_s = hll_flux(Q_L_s, Q_R_s, zb_interface)
        flux_grad_s = (F_int_s[1:] - F_int_s[:-1]) / dx

        F_b_L_s, F_b_R_s = barrier_face_fluxes(
            Q_L_s[barrier_idx, 0], Q_R_s[barrier_idx, 0], zb_interface[barrier_idx, 0]
        )
        flux_grad_s[barrier_idx - 1] = (F_b_L_s - F_int_s[barrier_idx - 1]) / dx
        flux_grad_s[barrier_idx]     = (F_int_s[barrier_idx + 1] - F_b_R_s) / dx

        S_s = get_source(Q_array, zb, dx, mannings_function)
        K2 = np.zeros_like(Q_array)
        K2[1:-1] = -flux_grad_s + S_s[1:-1]

        Q_array[:] = Q_n + 0.5 * dt * (K1 + K2)
        apply_bcs(Q_array, flow_rate, zb)

        max_change = np.max(np.abs(Q_array[:, 0] - Q_n[:, 0])) / dt

        if step % plot_freq == 0:
            eta = Q_array[:, 0]
            h = np.maximum(eta - zb[:, 0], 0.0)
            u = np.divide(Q_array[:, 1], h, out=np.zeros_like(h), where=h > 1e-4)
            line_eta.set_ydata(eta)
            line_u.set_ydata(u)
            ax1.relim(); ax1.autoscale_view()
            ax2.relim(); ax2.autoscale_view()
            ax1.set_title(f"Time: {t:.2f} s | Max Change: {max_change:.6f}")
            fig.canvas.draw()
            fig.canvas.flush_events()

        step += 1

    plt.ioff()
    plt.close()

    return t, Q_array