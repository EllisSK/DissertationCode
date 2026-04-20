import numpy as np
from time import sleep

from typing import Callable

def run_simulation(flow_rate: float, barrier_fn: Callable, bed_function: Callable | None = None):
    length = 12.5
    resolution = 0.1
    N = int(length / resolution)
    dx = resolution
    g = 9.81

    Barrier_idx = int(N / 2)

    def mannings_fn(h):
        return 0.0

    def q_t(t):
        return flow_rate

    Q_array = np.zeros((N + 2, 2))
    x_vals = np.linspace(dx / 2, length - (dx / 2), N)
    Q_array[1:-1] = 0.8
    Q_array[1:-1, 1] = flow_rate

    if bed_function:
        x_vals_full = np.concatenate(([dx / 2], x_vals, [length - (dx / 2)]))
        zb = bed_function(x_vals_full).reshape((N + 2, 1))
        zb_interface = 0.5 * (zb[:-1] + zb[1:])
    else:
        zb = np.zeros((N + 2, 1))
        zb_interface = np.zeros((N + 1, 1))

    def apply_boundaries(Q, current_t):
        Q[0, :] = Q[1, :]
        Q[0, 1] = q_t(current_t)
        Q[-1, :] = Q[-2, :]

    def get_flux(Q, z):
        eta = Q[:, 0]
        q = Q[:, 1]
        z_flat = z[:, 0]
        h = np.maximum(eta - z_flat, 0.0)
        u = np.divide(q, h, out=np.zeros_like(q), where=h > 0)
        F = np.zeros_like(Q)
        F[:, 0] = q
        F[:, 1] = (q * u) + (0.5 * g * (np.power(eta, 2) - (2 * eta * z_flat)))
        return F

    def get_source(Q):
        eta = Q[:, 0]
        q = Q[:, 1]
        h = np.maximum(eta - zb[:, 0], 0.0)
        dzb_dx = np.gradient(zb[:, 0], dx)
        bed_slope = -g * eta * dzb_dx

        mannings_n = mannings_fn(h)

        if mannings_n != 0:
            h_friction = np.maximum(h, 1e-3)
            friction = np.divide(g * (mannings_n**2) * q * np.abs(q), h_friction**(7/3), out=np.zeros_like(q), where=h > 1e-6)
        else:
            friction = np.zeros_like(bed_slope)

        S = np.zeros_like(Q)
        S[:, 1] = bed_slope - friction

        return S

    def reconstruct_spatial(Q):
        grad_U = np.zeros_like(Q)
        diff_bwd = (Q[1:-1] - Q[:-2]) / dx
        diff_fwd = (Q[2:] - Q[1:-1]) / dx
        grad_U[1:-1] = 0.5 * (np.sign(diff_bwd) + np.sign(diff_fwd)) * np.minimum(np.abs(diff_bwd), np.abs(diff_fwd))
        Q_L = Q[:-1] + (0.5 * dx * grad_U[:-1])
        Q_R = Q[1:] - (0.5 * dx * grad_U[1:])
        return Q_L, Q_R

    def solve_riemann(Q_L, Q_R):
        eta_L = Q_L[:, 0]
        q_L = Q_L[:, 1]
        eta_R = Q_R[:, 0]
        q_R = Q_R[:, 1]
        z_int = zb_interface[:, 0]

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

    t = 0.0
    max_change = 1.0
    convergence_threshold = 1e-6

    apply_boundaries(Q_array, t)

    while max_change > convergence_threshold:
        Q_n = Q_array.copy()

        h_dt = np.maximum(Q_array[:, 0] - zb[:, 0], 0.0)
        u_dt = np.divide(Q_array[:, 1], h_dt, out=np.zeros_like(Q_array[:, 1]), where=h_dt > 0.0)
        a_dt = np.sqrt(g * h_dt)
        max_speed = np.max(np.abs(u_dt) + a_dt)
        dt = np.nextafter(dx / max_speed, -np.inf)

        t += dt

        Q_L, Q_R = reconstruct_spatial(Q_array)
        F_int = solve_riemann(Q_L, Q_R)

        q_barrier = barrier_fn(Q_L[Barrier_idx][0], Q_R[Barrier_idx][0])
        Q_barrier = Q_L[Barrier_idx:Barrier_idx+1].copy()
        Q_barrier[0, 1] = q_barrier
        z_barrier = zb_interface[Barrier_idx:Barrier_idx+1]
        F_int[Barrier_idx] = get_flux(Q_barrier, z_barrier)[0]

        flux_grad = (F_int[1:] - F_int[:-1]) / dx
        S = get_source(Q_array)

        K1 = np.zeros_like(Q_array)
        K1[1:-1] = -flux_grad + S[1:-1]

        U_star = Q_array + (dt * K1)

        Q_array[:] = U_star
        apply_boundaries(Q_array, t)

        Q_L_star, Q_R_star = reconstruct_spatial(Q_array)
        F_int_star = solve_riemann(Q_L_star, Q_R_star)

        q_barrier_star = barrier_fn(Q_L_star[Barrier_idx, 0], Q_R_star[Barrier_idx, 0])
        Q_barrier_star = Q_L_star[Barrier_idx:Barrier_idx+1].copy()
        Q_barrier_star[0, 1] = q_barrier_star
        z_barrier_star = zb_interface[Barrier_idx:Barrier_idx+1]
        F_int_star[Barrier_idx] = get_flux(Q_barrier_star, z_barrier_star)[0]

        flux_grad_star = (F_int_star[1:] - F_int_star[:-1]) / dx
        S_star = get_source(Q_array)

        K2 = np.zeros_like(Q_array)
        K2[1:-1] = -flux_grad_star + S_star[1:-1]

        Q_array[:] = Q_n + 0.5 * dt * (K1 + K2)
        apply_boundaries(Q_array, t)

        max_change = np.max(np.abs(Q_array[:, 0] - Q_n[:, 0]))

        if t > 600:
            break

    return {t: Q_array}