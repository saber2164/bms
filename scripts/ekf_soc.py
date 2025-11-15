#!/usr/bin/env python3
"""
EKF_SoC_Estimator

Extended Kalman Filter (EKF) implementation for State-of-Charge (SoC)
estimation using a first-order equivalent circuit model (ECM / Thevenin).

State vector x = [SoC, U_p]^T where:
 - SoC: state of charge (0..1)
 - U_p: polarization (RC) voltage (V)

Process model (discrete-time):
 SoC_{k+1} = SoC_k - (dt / (3600 * C_nom)) * I_k / eta
 U_p_{k+1} = alpha * U_p_k + beta * I_k
 where alpha = exp(-dt / (R_D * C_D))
 and beta = R_D * (1 - alpha)

Observation model:
 V_k = OCV(SoC_k) - I_k * R0 - U_p_k + v_meas_noise

This file provides a clean, commented EKF class with Jacobians computed
analytically for the given model.

IMPORTANT: This EKF assumes a fixed sampling rate dt=1.0s. If your actual
measurement interval differs, adjust dt in the EKF constructor.

3-PARAMETER MODE (V, I, T only):
 - Input columns required: Voltage_measured, Current_measured, Temperature_measured
 - The EKF automatically computes: SoC via Coulomb counting, U_p via RC model
 - Temperature is logged but not used in model (for future thermal model integration)
 - No Time column needed; assumes uniform 1s sampling intervals

Author: automated assistant
"""
from typing import Optional, Sequence
import numpy as np
import pandas as pd
from math import exp


def fit_ocv_poly(file_paths: Sequence[str], C_nom: float, degree: int = 3,
                  current_threshold: float = 0.1, max_points: int = 20000) -> np.ndarray:
    """Fit a polynomial OCV(SOC) from a list of timeseries CSV files.

    Strategy:
    - For each file, compute an approximate SoC trace by Coulomb counting
      starting from SoC=1.0 (assumes files start near full SOC or relative scale).
    - Collect pairs (SoC, Voltage_measured) when |I| < current_threshold (rest/low-current).
    - Fit polynomial V = p(SOC) of degree `degree` using the pooled points.

    Returns polynomial coefficients (highest degree first) suitable for numpy.polyval.
    Falls back to a simple linear fit if insufficient data or fit fails.
    """
    xs = []
    ys = []
    for p in file_paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        if "Voltage_measured" not in df.columns or "Current_measured" not in df.columns:
            continue

        # Estimate time deltas
        if "Time" in df.columns:
            t = pd.to_numeric(df["Time"], errors="coerce")
            dt = t.diff().fillna(0).to_numpy()
            # any non-positive dt -> assume 1s
            dt = np.where(dt <= 0, 1.0, dt)
        else:
            dt = np.ones(len(df))

        I = pd.to_numeric(df["Current_measured"], errors="coerce").fillna(0).to_numpy()
        V = pd.to_numeric(df["Voltage_measured"], errors="coerce").fillna(np.nan).to_numpy()

        # cumulative charge in Ah (assuming current in A and dt in seconds)
        dq_ah = np.cumsum(I * dt) / 3600.0
        # rough SoC estimate assuming discharge reduces SoC
        soc_est = 1.0 - (dq_ah / max(1e-6, float(C_nom)))
        soc_est = np.clip(soc_est, 0.0, 1.0)

        mask = np.isfinite(V) & np.isfinite(soc_est) & (np.abs(I) <= current_threshold)
        # if too few low-current points, relax threshold to include more
        if np.sum(mask) < 20:
            mask = np.isfinite(V) & np.isfinite(soc_est)

        xs.extend(soc_est[mask].tolist())
        ys.extend(V[mask].tolist())

        if len(xs) >= max_points:
            break

    if len(xs) < 10:
        # not enough data; return a linear default (match previous placeholder)
        # default OCV(soc) = 3.4 + 0.8*SoC
        return np.array([0.8, 3.4])

    try:
        coeffs = np.polyfit(np.array(xs), np.array(ys), deg=degree)
        return coeffs
    except Exception:
        # fallback linear
        coeffs = np.polyfit(np.array(xs), np.array(ys), deg=1)
        return coeffs


class EKF_SoC_Estimator:
    """Extended Kalman Filter for SoC estimation using a 1st-order ECM.

    Parameters
    ----------
    dt : float
        Sampling time in seconds.
    C_nom : float
        Nominal capacity in Ah.
    R0 : float
        Ohmic (series) resistance in Ohm.
    R_D : float
        Polarization resistance (Ohm) for RC branch.
    C_D : float
        Polarization capacitance (Farad) for RC branch.
    eta : float
        Coulombic efficiency (charge/discharge efficiency), dimensionless.
    Q : optional np.ndarray
        Process noise covariance (2x2). If None, a small default is used.
    R : float
        Measurement noise variance (scalar). If None, default small value used.
    """

    def __init__(self, dt: float, C_nom: float, R0: float, R_D: float, C_D: float, eta: float,
                 Q: Optional[np.ndarray] = None, R: Optional[float] = None,
                 ocv_coeffs: Optional[Sequence[float]] = None,
                 max_soc_step: float = 0.1):
        self.dt = float(dt)
        self.C_nom = float(C_nom)
        self.R0 = float(R0)
        self.R_D = float(R_D)
        self.C_D = float(C_D)
        self.eta = float(eta)

        # polynomial coefficients for OCV(soc) if provided (highest power first)
        self.ocv_coeffs = None if ocv_coeffs is None else np.array(ocv_coeffs, dtype=float)

        # state x = [SoC, U_p]^T
        self.x = np.array([0.9, 0.0], dtype=float)  # initial guess: 90% SoC, zero polarization

        # covariance
        self.P = np.diag([1e-4, 1e-3])

        # process/measurement noise
        if Q is None:
            # small process noise on SoC and slightly larger for U_p drift
            self.Q = np.diag([1e-7, 1e-5])
        else:
            self.Q = np.array(Q, dtype=float)

        self.R = 1e-3 if R is None else float(R)

        # maximum allowed absolute SoC change in a single measurement update
        # helps avoid numerical/measurement outlier-driven collapse to 0/1
        self.max_soc_step = float(max_soc_step)

        # constant derived parameters
        self.tau = self.R_D * self.C_D if self.R_D * self.C_D > 0 else 1e-6
        self.alpha = exp(-self.dt / self.tau)
        # exact discrete step response coefficient
        self.beta = self.R_D * (1.0 - self.alpha)

    # --- OCV model (nonlinear) and its derivative ---
    def _ocv_model(self, soc: float) -> float:
        """Simple placeholder OCV model: OCV(soc).

        Replace this with a fitted OCV curve or lookup table in real use.
        """
        if self.ocv_coeffs is not None and len(self.ocv_coeffs) >= 2:
            # evaluate polynomial
            return float(np.polyval(self.ocv_coeffs, float(soc)))
        # simple linear placeholder (V): OCV = 3.4 + 0.8 * SoC
        return 3.4 + 0.8 * float(soc)

    def _ocv_derivative(self, soc: float) -> float:
        """Derivative dOCV/dSoC for the placeholder model."""
        if self.ocv_coeffs is not None and len(self.ocv_coeffs) >= 2:
            # derivative polynomial coefficients
            der = np.polyder(self.ocv_coeffs)
            return float(np.polyval(der, float(soc)))
        return 0.8

    # --- state transition and observation functions ---
    def _state_transition_function(self, x: np.ndarray, i_k: float) -> np.ndarray:
        """Compute f(x, u) -- discrete-time state transition.

        x: [SoC, U_p]
        i_k: current (A), positive for charge
        """
        soc, u_p = float(x[0]), float(x[1])
        # Coulomb counting (SoC change). Convert C_nom in Ah to Coulombs via 3600.
        dsoc = - (self.dt * i_k) / (3600.0 * self.C_nom * self.eta)
        soc_next = soc + dsoc

        # polarization RC exact discrete update
        u_p_next = self.alpha * u_p + self.beta * i_k

        return np.array([soc_next, u_p_next], dtype=float)

    def _observation_function(self, x: np.ndarray, i_k: float) -> float:
        """Observation model g(x, u): terminal voltage.

        V = OCV(SoC) - I * R0 - U_p
        """
        soc, u_p = float(x[0]), float(x[1])
        return float(self._ocv_model(soc) - i_k * self.R0 - u_p)

    # --- EKF steps ---
    def predict(self, i_k: float):
        """EKF predict step using input current i_k."""
        # compute Jacobian A = df/dx at current x
        A = np.array([[1.0, 0.0],
                      [0.0, self.alpha]], dtype=float)

        # predict state
        self.x = self._state_transition_function(self.x, i_k)

        # propagate covariance
        self.P = A @ self.P @ A.T + self.Q

        # constrain SoC to [0,1]
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)

        return self.x.copy()

    def update(self, v_measured: float, i_k: float):
        """EKF measurement update with measured terminal voltage v_measured.

        Returns updated state estimate.
        """
        # Jacobian of observation g wrt state x: C = dg/dx
        # dg/dSoC = dOCV/dSoC, dg/dU_p = -1
        dOCV_dSoC = self._ocv_derivative(self.x[0])
        C = np.array([[dOCV_dSoC, -1.0]], dtype=float)  # shape (1,2)

        # predicted observation
        y_pred = self._observation_function(self.x, i_k)
        y = float(v_measured) - y_pred

        # innovation covariance
        S = C @ self.P @ C.T + self.R  # scalar

        # Kalman gain
        K = (self.P @ C.T) / S  # shape (2,1)

        # update state and covariance
        new_x = self.x + (K.flatten() * y)
        # guard: limit SoC change to reasonable per-step magnitude to avoid collapse from outliers
        prior_soc = float(self.x[0])
        proposed_soc = float(new_x[0])
        delta = proposed_soc - prior_soc
        if np.isfinite(delta) and abs(delta) > self.max_soc_step:
            proposed_soc = prior_soc + np.sign(delta) * self.max_soc_step
        new_x[0] = np.clip(proposed_soc, 0.0, 1.0)
        self.x = new_x
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ C) @ self.P

        # clamp SoC
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)

        return self.x.copy()


if __name__ == "__main__":
    # Quick smoke test for the EKF implementation.
    ekf = EKF_SoC_Estimator(dt=1.0, C_nom=2.3, R0=0.05, R_D=0.01, C_D=500.0, eta=0.99)

    # simulate a short discharge: constant -1 A for a few seconds
    true_soc = 0.95
    true_up = 0.0
    I = -1.0

    print("Initial est:", ekf.x)
    for k in range(10):
        # simulate true plant (simple)
        true_soc = np.clip(true_soc - (ekf.dt * I) / (3600.0 * ekf.C_nom * ekf.eta), 0.0, 1.0)
        true_up = ekf.alpha * true_up + ekf.beta * I
        v_meas = ekf._ocv_model(true_soc) - I * ekf.R0 - true_up + np.random.randn() * 1e-3

        ekf.predict(I)
        x_upd = ekf.update(v_meas, I)
        print(f"k={k:02d}, measV={v_meas:.3f}, estSoC={x_upd[0]:.4f}, estUp={x_upd[1]:.4f}")
