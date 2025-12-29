#!/usr/bin/env python3
"""
Refactored Square-Root Dual Unscented Kalman Filter (SR-DUKF).

Improvements:
1.  **Vectorized Observation**: Predicts OCV for all sigma points in a single batch call, significantly improving speed.
2.  **Scaler Integration**: Applies the saved MinMaxScaler to inputs before passing them to the OCV model.
3.  **Robustness**: Added checks for positive parameters.
"""

import numpy as np
from scipy.linalg import cholesky, qr

class DualUKF:
    def __init__(self, dt, C_nom, R0_nom, ocv_model, scaler, alpha=1e-3, beta=2., kappa=0.):
        self.dt = dt
        self.ocv_model = ocv_model
        self.scaler = scaler # MinMaxScaler for OCV inputs

        # --- Filter Tuning Parameters ---
        # Process noise
        self.q_state = np.diag([1e-6, 1e-6]) 
        self.q_param = np.diag([1e-8, 1e-9])
        # Measurement noise
        self.r_state = np.array([[1e-2]])
        self.r_param = np.array([[1e-2]])

        # --- State Filter (UKF) Initialization ---
        self.x = np.array([0.9, 0.0])  # State: [SoC, U_d]
        self.n_x = len(self.x)
        self.S_x = cholesky(np.diag([1e-4, 1e-4])) 

        # --- Parameter Filter (UKF) Initialization ---
        self.theta = np.array([C_nom, R0_nom]) # Parameters: [Q_max, R_0]
        self.n_theta = len(self.theta)
        self.S_theta = cholesky(np.diag([1e-2, 1e-4]))

        # --- UKF Sigma Point Generation Parameters ---
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_x = alpha**2 * (self.n_x + kappa) - self.n_x
        self.lambda_theta = alpha**2 * (self.n_theta + kappa) - self.n_theta
        self.W_m_x, self.W_c_x = self._compute_weights(self.n_x, self.lambda_x)
        self.W_m_theta, self.W_c_theta = self._compute_weights(self.n_theta, self.lambda_theta)

        # --- Battery Model Parameters ---
        self.R_D = 0.01
        self.C_D = 1000
        self.eta = 0.99
        
        self.step_counter = 0
        self.parameter_filter_update_rate = 100

    def _compute_weights(self, n, lambda_):
        W_m = np.full(2 * n + 1, 1. / (2. * (n + lambda_)))
        W_c = np.full(2 * n + 1, 1. / (2. * (n + lambda_)))
        W_m[0] = lambda_ / (n + lambda_)
        W_c[0] = lambda_ / (n + lambda_) + (1. - self.alpha**2 + self.beta)
        return W_m, W_c

    def _generate_sigma_points(self, x, S, lambda_):
        n = len(x)
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = x
        gamma = np.sqrt(n + lambda_)
        for i in range(n):
            sigma_points[i + 1]   = x + gamma * S[:, i]
            sigma_points[i + 1 + n] = x - gamma * S[:, i]
        return sigma_points

    def _state_transition(self, x, i_k):
        soc_k, u_d_k = x
        Q_max = self.theta[0]
        
        # Coulomb Counting for SoC
        # i_k > 0 is Discharge (SoC decreases)
        soc_next = soc_k - (self.eta * i_k / (Q_max * 3600)) * self.dt
        
        # RC Pair Dynamics
        exp_val = np.exp(-self.dt / (self.R_D * self.C_D))
        u_d_next = u_d_k * exp_val + self.R_D * (1 - exp_val) * i_k
        
        return np.array([soc_next, u_d_next])

    def _batch_observation(self, sigma_points_x, theta, temp_k, i_k):
        """
        Vectorized observation function.
        Calculates predicted voltage for ALL sigma points at once.
        """
        R_0 = theta[1]
        
        # Extract SoC from all sigma points
        soc_values = sigma_points_x[:, 0]
        u_d_values = sigma_points_x[:, 1]
        
        # Prepare inputs for OCV model: [SoC, Temp]
        # Create a column of temperature values matching the number of sigma points
        temp_values = np.full_like(soc_values, temp_k)
        
        # Stack into (N, 2) array
        model_inputs = np.column_stack((soc_values, temp_values))
        
        # Scale inputs using the saved scaler
        model_inputs_scaled = self.scaler.transform(model_inputs)
        
        # Batch Predict OCV
        # verbose=0 to suppress progress bar
        ocv_preds = self.ocv_model.predict(model_inputs_scaled, verbose=0).flatten()
        
        # Calculate Terminal Voltage: V = OCV - I*R0 - U_d
        # Assuming Discharge Current I > 0 reduces voltage
        v_preds = ocv_preds - i_k * R_0 - u_d_values
        
        return v_preds

    def step(self, v_meas, i_k, temp_k):
        # --- State Estimation ---
        sigma_x = self._generate_sigma_points(self.x, self.S_x, self.lambda_x)
        
        # Propagate State
        sigma_x_pred = np.array([self._state_transition(s, i_k) for s in sigma_x])
        x_pred = np.dot(self.W_m_x, sigma_x_pred)
        
        X = sigma_x_pred - x_pred[np.newaxis, :]
        R = qr(np.sqrt(self.W_c_x[1]) * X[1:, :].T)[0].T
        S_x_pred = cholesky(R @ R.T + self.q_state, lower=False)

        # Propagate Measurement (Vectorized)
        # We pass the PREDICTED sigma points to the observation model
        sigma_y_pred = self._batch_observation(sigma_x_pred, self.theta, temp_k, i_k)
        y_pred = np.dot(self.W_m_x, sigma_y_pred)
        
        Y = (sigma_y_pred - y_pred)[:, np.newaxis]
        P_yy = (Y.T @ np.diag(self.W_c_x) @ Y) + self.r_state
        P_xy = X.T @ np.diag(self.W_c_x) @ Y
        
        K = P_xy / P_yy
        
        # Update State
        self.x = x_pred + K.flatten() * (v_meas - y_pred)
        U = K @ cholesky(P_yy, lower=False)
        self.S_x = cholesky(S_x_pred.T @ S_x_pred - U @ U.T, lower=False)
        self.x[0] = np.clip(self.x[0], 0, 1)

        # --- Parameter Estimation ---
        if self.step_counter % self.parameter_filter_update_rate == 0:
            sigma_theta = self._generate_sigma_points(self.theta, self.S_theta, self.lambda_theta)
            
            theta_pred = np.dot(self.W_m_theta, sigma_theta)
            Theta = sigma_theta - theta_pred[np.newaxis, :]
            R_theta = qr(np.sqrt(self.W_c_theta[1]) * Theta[1:, :].T)[0].T
            S_theta_pred = cholesky(R_theta @ R_theta.T + self.q_param, lower=False)

            # Observation for parameters
            # We need to vary theta (from sigma_theta) but keep state fixed (self.x)
            # To use _batch_observation, we need to construct a "sigma_x" that just repeats self.x
            # but that's not quite right because _batch_observation takes sigma_x and SINGLE theta.
            
            # We'll loop here because vectorizing over parameters is trickier with the current helper structure
            # and parameter update is infrequent (1/100 steps), so performance hit is negligible.
            sigma_y_theta = []
            for s_theta in sigma_theta:
                # Create a single-row "sigma_x" containing the current state estimate
                # This is a bit hacky but reuses the logic
                single_state = self.x.reshape(1, -1)
                pred = self._batch_observation(single_state, s_theta, temp_k, i_k)
                sigma_y_theta.append(pred[0])
            
            sigma_y_theta = np.array(sigma_y_theta)
            
            y_theta_pred = np.dot(self.W_m_theta, sigma_y_theta)
            
            Y_theta = (sigma_y_theta - y_theta_pred)[:, np.newaxis]
            P_yy_theta = (Y_theta.T @ np.diag(self.W_c_theta) @ Y_theta) + self.r_param
            P_theta_y = Theta.T @ np.diag(self.W_c_theta) @ Y_theta
            
            K_theta = P_theta_y / P_yy_theta
            
            self.theta = theta_pred + K_theta.flatten() * (v_meas - y_theta_pred)
            U_theta = K_theta @ cholesky(P_yy_theta, lower=False)
            self.S_theta = cholesky(S_theta_pred.T @ S_theta_pred - U_theta @ U_theta.T, lower=False)

            # Constraints
            self.theta[0] = max(self.theta[0], 0.1) # Min Capacity
            self.theta[1] = max(self.theta[1], 1e-4) # Min Resistance

        self.step_counter += 1
        return self.x, self.theta
