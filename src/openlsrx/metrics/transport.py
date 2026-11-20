"""3D pore-scale diffusive tortuosity and capillary drainage physics metrics for XCT."""

from __future__ import annotations

from typing import Any

import numpy as np
import porespy as ps
from scipy import sparse
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label
from scipy.sparse.linalg import LinearOperator, cg, spsolve

from .petrophysics import ThresholdMethod, segment_pore_space


def solve_steady_state_diffusion_axis(
    pore_mask: np.ndarray,
    axis: int = 0,
    tol: float = 1e-6,
    maxiter: int = 4000,
) -> dict[str, float]:
    """Solve 3D steady-state Fickian Laplace diffusion ``div(grad c) = 0`` along ``axis``.

    Verifies solver convergence (both ``cg`` exit status and inlet/outlet steady-state flux
    conservation), falling back to exact sparse LU (when ``n_nodes <= 40_000``) or extended
    high-precision CG when ill-conditioned constrictions stall iterative convergence.
    """
    mask = np.asarray(pore_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D boolean pore_mask, got shape {mask.shape}.")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}.")

    porosity = float(mask.mean())
    if porosity == 0.0:
        return {
            "porosity": 0.0,
            "percolating_porosity": 0.0,
            "d_eff": 0.0,
            "tortuosity": float("inf"),
            "formation_factor": float("inf"),
            "percolates": 0.0,
            "solver_converged": 1.0,
        }

    ordered = np.moveaxis(mask, axis, 0)  # (L, M, N)
    length, m_dim, n_dim = ordered.shape
    a_geom = float(m_dim * n_dim)

    conn6 = generate_binary_structure(3, 1)
    labels, _ = label(ordered, structure=conn6)
    inlet_labels = set(np.unique(labels[0])) - {0}
    outlet_labels = set(np.unique(labels[-1])) - {0}
    perc_labels = inlet_labels & outlet_labels

    if not perc_labels:
        return {
            "porosity": porosity,
            "percolating_porosity": 0.0,
            "d_eff": 0.0,
            "tortuosity": float("inf"),
            "formation_factor": float("inf"),
            "percolates": 0.0,
            "solver_converged": 1.0,
        }

    perc_mask = np.isin(labels, list(perc_labels))
    perc_porosity = float(perc_mask.mean())

    if length == 1:
        d_eff = perc_porosity
        return {
            "porosity": porosity,
            "percolating_porosity": perc_porosity,
            "d_eff": d_eff,
            "tortuosity": porosity / d_eff,
            "formation_factor": 1.0 / d_eff,
            "percolates": 1.0,
            "solver_converged": 1.0,
        }

    active_flat = np.flatnonzero(perc_mask)
    n_nodes = int(active_flat.size)
    node_idx = np.full(perc_mask.shape, -1, dtype=np.int64)
    node_idx.ravel()[active_flat] = np.arange(n_nodes, dtype=np.int64)

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    diag = np.zeros(n_nodes, dtype=np.float64)

    for dim in range(3):
        sl_lo = [slice(None), slice(None), slice(None)]
        sl_hi = [slice(None), slice(None), slice(None)]
        sl_lo[dim] = slice(0, -1)
        sl_hi[dim] = slice(1, None)
        both = perc_mask[tuple(sl_lo)] & perc_mask[tuple(sl_hi)]
        u = node_idx[tuple(sl_lo)][both]
        v = node_idx[tuple(sl_hi)][both]
        if u.size > 0:
            rows_list.append(u)
            cols_list.append(v)
            rows_list.append(v)
            cols_list.append(u)
            np.add.at(diag, u, 1.0)
            np.add.at(diag, v, 1.0)

    rhs = np.zeros(n_nodes, dtype=np.float64)
    inlet_nodes = node_idx[0][perc_mask[0]]
    outlet_nodes = node_idx[-1][perc_mask[-1]]
    diag[inlet_nodes] += 2.0
    rhs[inlet_nodes] += 2.0
    diag[outlet_nodes] += 2.0

    diag_idx = np.arange(n_nodes, dtype=np.int64)
    if rows_list:
        off_r = np.concatenate(rows_list)
        off_c = np.concatenate(cols_list)
        off_v = np.full(off_r.size, -1.0, dtype=np.float64)
        all_r = np.concatenate([diag_idx, off_r])
        all_c = np.concatenate([diag_idx, off_c])
        all_v = np.concatenate([diag, off_v])
    else:
        all_r = diag_idx
        all_c = diag_idx
        all_v = diag

    mat = sparse.csr_matrix((all_v, (all_r, all_c)), shape=(n_nodes, n_nodes))
    inv_diag = 1.0 / np.maximum(diag, 1e-12)
    precond = LinearOperator(
        shape=(n_nodes, n_nodes),
        dtype=np.float64,
        matvec=lambda x: inv_diag * x,
    )
    conc, info = cg(mat, rhs, rtol=tol, atol=0.0, maxiter=maxiter, M=precond)

    def _flux_metrics(c_vec: np.ndarray) -> tuple[float, float, float, float]:
        qi = float(np.sum(2.0 * (1.0 - c_vec[inlet_nodes])))
        qo = float(np.sum(2.0 * c_vec[outlet_nodes]))
        qm = max(0.0, 0.5 * (qi + qo))
        imbalance = abs(qi - qo) / max(qm, 1e-12)
        return qi, qo, qm, imbalance

    _, _, q_mean, flux_imbalance = _flux_metrics(conc)
    rhs_norm = max(float(np.linalg.norm(rhs)), 1e-12)
    rel_res = float(np.linalg.norm(mat @ conc - rhs)) / rhs_norm
    converged = (info == 0) and (flux_imbalance <= 0.02) and (rel_res <= max(tol * 50.0, 1e-4))

    if not converged:
        if n_nodes <= 40_000:
            conc = np.asarray(spsolve(mat.tocsc(), rhs), dtype=np.float64)
        else:
            conc, info = cg(
                mat,
                rhs,
                x0=np.clip(conc, 0.0, 1.0),
                rtol=min(tol, 1e-8),
                atol=0.0,
                maxiter=max(maxiter * 5, 20_000),
                M=precond,
            )
        conc = np.clip(conc, 0.0, 1.0)
        _, _, q_mean, flux_imbalance = _flux_metrics(conc)
        rel_res = float(np.linalg.norm(mat @ conc - rhs)) / rhs_norm
        converged = (flux_imbalance <= 0.05) and (rel_res <= 1e-3)

    # Variational Dirichlet bound: 0 <= D_eff <= percolating_porosity <= porosity, so tau >= 1.0
    d_eff = float(np.clip((q_mean * float(length)) / a_geom, 0.0, perc_porosity))
    if d_eff <= 1e-12 or not np.isfinite(d_eff):
        return {
            "porosity": porosity,
            "percolating_porosity": perc_porosity,
            "d_eff": 0.0,
            "tortuosity": float("inf"),
            "formation_factor": float("inf"),
            "percolates": 0.0,
            "solver_converged": float(converged),
        }

    return {
        "porosity": porosity,
        "percolating_porosity": perc_porosity,
        "d_eff": d_eff,
        "tortuosity": max(1.0, porosity / d_eff),
        "formation_factor": 1.0 / d_eff,
        "percolates": 1.0,
        "solver_converged": float(converged),
    }


def compute_directional_tortuosity(
    pore_mask: np.ndarray,
    tol: float = 1e-6,
    maxiter: int = 4000,
) -> dict[str, float]:
    """Compute directional effective diffusivity ``D_eff/D_0``, tortuosity ``tau``, and anisotropy across ``z, y, x``.

    Reports ``tortuosity_mean = porosity / d_eff_mean`` (the macroscopic 3D tensor-trace bulk
    tortuosity ``phi / ((D_z + D_y + D_x) / 3)``), which equals directional ``tau`` for isotropic
    media and varies monotonically across single-axis percolation transitions without dropping
    disconnected ``inf`` axes.
    """
    res_z = solve_steady_state_diffusion_axis(pore_mask, axis=0, tol=tol, maxiter=maxiter)
    res_y = solve_steady_state_diffusion_axis(pore_mask, axis=1, tol=tol, maxiter=maxiter)
    res_x = solve_steady_state_diffusion_axis(pore_mask, axis=2, tol=tol, maxiter=maxiter)

    d_vals = [res_z["d_eff"], res_y["d_eff"], res_x["d_eff"]]
    d_mean = float(np.mean(d_vals))
    tau_mean = max(1.0, res_z["porosity"] / d_mean) if d_mean > 1e-12 else float("inf")
    finite_taus = [r["tortuosity"] for r in (res_z, res_y, res_x) if np.isfinite(r["tortuosity"])]
    tau_arith_perc = float(np.mean(finite_taus)) if len(finite_taus) == 3 else float("inf")
    ff_mean = (1.0 / d_mean) if d_mean > 1e-12 else float("inf")

    min_d = min(d_vals)
    max_d = max(d_vals)
    aniso_ratio = (max_d / min_d) if min_d > 1e-12 else (1.0 if max_d <= 1e-12 else float("inf"))

    return {
        "porosity": res_z["porosity"],
        "d_eff_z": res_z["d_eff"],
        "d_eff_y": res_y["d_eff"],
        "d_eff_x": res_x["d_eff"],
        "d_eff_mean": d_mean,
        "tortuosity_z": res_z["tortuosity"],
        "tortuosity_y": res_y["tortuosity"],
        "tortuosity_x": res_x["tortuosity"],
        "tortuosity_mean": tau_mean,
        "tortuosity_3axis_arithmetic": tau_arith_perc,
        "formation_factor_mean": ff_mean,
        "percolating_axes_count": res_z["percolates"] + res_y["percolates"] + res_x["percolates"],
        "d_eff_anisotropy_spread": max_d - min_d,
        "transport_anisotropy_ratio": aniso_ratio,
        "solver_converged": float(
            min(res_z["solver_converged"], res_y["solver_converged"], res_x["solver_converged"])
        ),
    }


def compute_drainage_curve(
    pore_mask: np.ndarray,
    n_steps: int = 16,
    voxel_size: float = 1.0,
    inlet_axis: int = 0,
    pc_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate inlet-connected capillary drainage (mercury intrusion) on a 3D pore mask."""
    mask = np.asarray(pore_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D boolean pore_mask, got shape {mask.shape}.")

    n_pores = int(mask.sum())
    if n_pores == 0:
        grid = (
            np.geomspace(0.1, 2.0, n_steps, dtype=np.float64)
            if pc_grid is None
            else np.asarray(pc_grid, dtype=np.float64)
        )
        return grid, np.zeros_like(grid, dtype=np.float64)

    dt = distance_transform_edt(mask) * float(voxel_size)
    if pc_grid is None:
        r_vals = dt[mask]
        r_min = max(float(r_vals.min()), 0.5 * float(voxel_size))
        r_max = max(float(r_vals.max()), r_min * 1.05)
        grid = np.geomspace(2.0 / r_max, 2.0 / r_min, n_steps, dtype=np.float64)
    else:
        grid = np.asarray(pc_grid, dtype=np.float64)

    inlets = np.zeros_like(mask, dtype=bool)
    sl: list[slice | int] = [slice(None), slice(None), slice(None)]
    sl[inlet_axis] = 0
    inlets[tuple(sl)] = mask[tuple(sl)]
    if not inlets.any():
        return grid, np.zeros_like(grid, dtype=np.float64)

    pc_transform = np.full(mask.shape, np.inf, dtype=np.float32)
    pc_transform[mask] = 2.0 / np.maximum(dt[mask], 1e-6).astype(np.float32)

    drn = ps.simulations.drainage(
        im=mask,
        pc=pc_transform,
        dt=dt.astype(np.float32),
        inlets=inlets,
        steps=np.unique(np.sort(np.asarray(grid, dtype=np.float32))),
    )
    im_pc = np.asarray(drn.im_pc, dtype=np.float64)
    pore_pc = im_pc[mask]
    snwp = np.array([float(np.sum(pore_pc <= p)) / float(n_pores) for p in grid], dtype=np.float64)
    return grid, snwp


def compute_drainage_rmse(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    n_steps: int = 16,
    voxel_size: float = 1.0,
    inlet_axis: int = 0,
    precomputed_true_curve: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """RMSE between simulated capillary drainage saturation curves ``S_nwp(P_c)``."""
    if precomputed_true_curve is not None:
        pc_grid, true_snwp = precomputed_true_curve
    else:
        pc_grid, true_snwp = compute_drainage_curve(
            true_mask,
            n_steps=n_steps,
            voxel_size=voxel_size,
            inlet_axis=inlet_axis,
        )
    _, pred_snwp = compute_drainage_curve(
        pred_mask,
        n_steps=len(pc_grid),
        voxel_size=voxel_size,
        inlet_axis=inlet_axis,
        pc_grid=pc_grid,
    )
    return float(np.sqrt(np.mean((pred_snwp - true_snwp) ** 2)))


def precompute_reference_transport(
    true_vol: np.ndarray,
    threshold_method: ThresholdMethod = "otsu",
    fixed_threshold: float | None = None,
    pore_is_low: bool = True,
    voxel_size: float = 1.0,
    drainage_steps: int = 16,
    fringe_guard_alpha: float = 0.0,
) -> dict[str, Any]:
    """Precompute HR ground-truth threshold, 3D Laplace tortuosity, and drainage curve once per seed."""
    true_mask, thr = segment_pore_space(
        true_vol,
        method=threshold_method,
        fixed_value=fixed_threshold,
        pore_is_low=pore_is_low,
        fringe_guard_alpha=fringe_guard_alpha,
    )
    true_t = compute_directional_tortuosity(true_mask)
    true_curve = compute_drainage_curve(
        true_mask,
        n_steps=drainage_steps,
        voxel_size=voxel_size,
    )
    return {
        "threshold": float(thr),
        "fringe_guard_alpha": float(fringe_guard_alpha),
        "true_mask": true_mask,
        "true_tortuosity": true_t,
        "true_drainage_curve": true_curve,
    }


def _json_safe(val: float) -> float | None:
    return float(val) if np.isfinite(val) else None


def evaluate_transport_metrics(
    pred_vol: np.ndarray,
    true_vol: np.ndarray,
    threshold_method: ThresholdMethod = "otsu",
    fixed_threshold: float | None = None,
    pore_is_low: bool = True,
    voxel_size: float = 1.0,
    drainage_steps: int = 16,
    precomputed_true: dict[str, Any] | None = None,
    fringe_guard_alpha: float = 0.0,
) -> dict[str, float | None]:
    """Evaluate 3D diffusive transport and capillary drainage fidelity between SR and HR XCT volumes."""
    eff_alpha = float(fringe_guard_alpha)
    if precomputed_true is not None:
        thr = float(precomputed_true["threshold"])
        eff_alpha = float(precomputed_true.get("fringe_guard_alpha", eff_alpha))
        true_mask = np.asarray(precomputed_true["true_mask"], dtype=bool)
        true_t = dict(precomputed_true["true_tortuosity"])
        true_curve = precomputed_true.get("true_drainage_curve")
    else:
        true_mask, thr = segment_pore_space(
            true_vol,
            method=threshold_method,
            fixed_value=fixed_threshold,
            pore_is_low=pore_is_low,
            fringe_guard_alpha=eff_alpha,
        )
        true_t = compute_directional_tortuosity(true_mask)
        true_curve = None

    pred_mask, _ = segment_pore_space(
        pred_vol,
        method="fixed",
        fixed_value=thr,
        pore_is_low=pore_is_low,
        fringe_guard_alpha=eff_alpha,
    )
    pred_t = compute_directional_tortuosity(pred_mask)
    if float(pred_t["solver_converged"]) < 1.0 or float(true_t["solver_converged"]) < 1.0:
        import warnings

        warnings.warn(
            f"Steady-state diffusion solver reported incomplete convergence "
            f"(pred={pred_t['solver_converged']:.1f}, true={true_t['solver_converged']:.1f}).",
            UserWarning,
            stacklevel=2,
        )

    d_eff_mae = float(
        np.mean(
            [
                abs(pred_t["d_eff_z"] - true_t["d_eff_z"]),
                abs(pred_t["d_eff_y"] - true_t["d_eff_y"]),
                abs(pred_t["d_eff_x"] - true_t["d_eff_x"]),
            ]
        )
    )
    d_eff_rel_err = (
        abs(pred_t["d_eff_mean"] - true_t["d_eff_mean"]) / true_t["d_eff_mean"]
        if true_t["d_eff_mean"] > 1e-12
        else None
    )
    tau_abs_err = (
        abs(pred_t["tortuosity_mean"] - true_t["tortuosity_mean"])
        if np.isfinite(pred_t["tortuosity_mean"]) and np.isfinite(true_t["tortuosity_mean"])
        else None
    )
    drainage_rmse = compute_drainage_rmse(
        pred_mask,
        true_mask,
        n_steps=drainage_steps,
        voxel_size=voxel_size,
        precomputed_true_curve=true_curve,
    )

    return {
        "threshold_used": float(thr),
        "pred_d_eff_z": _json_safe(pred_t["d_eff_z"]),
        "pred_d_eff_y": _json_safe(pred_t["d_eff_y"]),
        "pred_d_eff_x": _json_safe(pred_t["d_eff_x"]),
        "pred_d_eff_mean": _json_safe(pred_t["d_eff_mean"]),
        "true_d_eff_mean": _json_safe(true_t["d_eff_mean"]),
        "d_eff_mae": d_eff_mae,
        "d_eff_rel_error": _json_safe(d_eff_rel_err) if d_eff_rel_err is not None else None,
        "pred_tortuosity_mean": _json_safe(pred_t["tortuosity_mean"]),
        "true_tortuosity_mean": _json_safe(true_t["tortuosity_mean"]),
        "tortuosity_abs_error": _json_safe(tau_abs_err) if tau_abs_err is not None else None,
        "pred_formation_factor_mean": _json_safe(pred_t["formation_factor_mean"]),
        "true_formation_factor_mean": _json_safe(true_t["formation_factor_mean"]),
        "pred_percolating_axes": float(pred_t["percolating_axes_count"]),
        "true_percolating_axes": float(true_t["percolating_axes_count"]),
        "pred_d_eff_anisotropy_spread": float(pred_t["d_eff_anisotropy_spread"]),
        "true_d_eff_anisotropy_spread": float(true_t["d_eff_anisotropy_spread"]),
        "pred_transport_anisotropy_ratio": _json_safe(pred_t["transport_anisotropy_ratio"]),
        "true_transport_anisotropy_ratio": _json_safe(true_t["transport_anisotropy_ratio"]),
        "pred_solver_converged": float(pred_t["solver_converged"]),
        "true_solver_converged": float(true_t["solver_converged"]),
        "drainage_snwp_rmse": drainage_rmse,
    }


def compute_tortuosity_uncertainty_bounds(
    pred_vol: np.ndarray,
    uncertainty_std: np.ndarray,
    true_vol: np.ndarray | None = None,
    threshold_method: ThresholdMethod = "otsu",
    fixed_threshold: float | None = None,
    pore_is_low: bool = True,
    k_sigma: float = 1.0,
    precomputed_true: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    """Propagate voxel-wise orthogonal view-disagreement ``sigma_ortho`` into 3D tortuosity bounds."""
    from scipy.stats import spearmanr

    pred_arr = np.asarray(pred_vol, dtype=np.float32)
    unc_arr = np.asarray(uncertainty_std, dtype=np.float32)
    if pred_arr.shape != unc_arr.shape or pred_arr.ndim != 3:
        raise ValueError(
            f"Expected matching 3D arrays for pred_vol and uncertainty_std, got {pred_arr.shape} and {unc_arr.shape}."
        )

    if precomputed_true is not None:
        thr = float(precomputed_true["threshold"])
    else:
        ref_for_thr = np.asarray(true_vol, dtype=np.float32) if true_vol is not None else pred_arr
        _, thr = segment_pore_space(
            ref_for_thr,
            method=threshold_method,
            fixed_value=fixed_threshold,
            pore_is_low=pore_is_low,
        )

    delta = float(k_sigma) * np.maximum(unc_arr, 0.0)
    vol_open = np.clip(pred_arr - delta if pore_is_low else pred_arr + delta, 0.0, 1.0)
    vol_tight = np.clip(pred_arr + delta if pore_is_low else pred_arr - delta, 0.0, 1.0)

    mask_nom, _ = segment_pore_space(
        pred_arr, method="fixed", fixed_value=thr, pore_is_low=pore_is_low
    )
    mask_open, _ = segment_pore_space(
        vol_open, method="fixed", fixed_value=thr, pore_is_low=pore_is_low
    )
    mask_tight, _ = segment_pore_space(
        vol_tight, method="fixed", fixed_value=thr, pore_is_low=pore_is_low
    )

    t_nom = compute_directional_tortuosity(mask_nom)
    t_open = compute_directional_tortuosity(mask_open)
    t_tight = compute_directional_tortuosity(mask_tight)

    d_candidates = [t_tight["d_eff_mean"], t_nom["d_eff_mean"], t_open["d_eff_mean"]]
    d_lo = float(min(d_candidates))
    d_hi = float(max(d_candidates))

    tau_candidates = [
        t_open["tortuosity_mean"],
        t_nom["tortuosity_mean"],
        t_tight["tortuosity_mean"],
    ]
    tau_lo = float(min(tau_candidates))
    tau_hi = float(max(tau_candidates))
    tau_hw = (
        0.5 * (tau_hi - tau_lo) if (np.isfinite(tau_lo) and np.isfinite(tau_hi)) else float("inf")
    )

    mean_std = float(unc_arr.mean())
    p95_std = float(np.percentile(unc_arr, 95.0))
    result: dict[str, float | None] = {
        "threshold_used": float(thr),
        "mean_ortho_std": mean_std,
        "p95_ortho_std": p95_std,
        "mean_epistemic_std": mean_std,
        "p95_epistemic_std": p95_std,
        "d_eff_nominal": _json_safe(t_nom["d_eff_mean"]),
        "d_eff_lower": _json_safe(d_lo),
        "d_eff_upper": _json_safe(d_hi),
        "d_eff_halfwidth": float(0.5 * (d_hi - d_lo)),
        "tortuosity_nominal": _json_safe(t_nom["tortuosity_mean"]),
        "tortuosity_lower": _json_safe(tau_lo),
        "tortuosity_upper": _json_safe(tau_hi),
        "tortuosity_halfwidth": _json_safe(tau_hw),
    }

    if true_vol is not None:
        true_arr = np.asarray(true_vol, dtype=np.float32)
        if precomputed_true is not None:
            t_true = dict(precomputed_true["true_tortuosity"])
        else:
            true_mask, _ = segment_pore_space(
                true_arr, method="fixed", fixed_value=thr, pore_is_low=pore_is_low
            )
            t_true = compute_directional_tortuosity(true_mask)
        d_true = float(t_true["d_eff_mean"])
        result["true_d_eff_mean"] = _json_safe(d_true)
        result["d_eff_bracketed_by_envelope"] = float(d_lo - 1e-8 <= d_true <= d_hi + 1e-8)

        abs_err = np.abs(pred_arr - true_arr).ravel()
        u_flat = unc_arr.ravel()
        if float(np.std(u_flat)) > 1e-12 and float(np.std(abs_err)) > 1e-12:
            corr_res = spearmanr(u_flat, abs_err)
            rho = float(corr_res.statistic)
            result["uncertainty_error_spearman_r"] = _json_safe(rho)
        else:
            result["uncertainty_error_spearman_r"] = 0.0

    return result
