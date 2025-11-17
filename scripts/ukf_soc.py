#!/usr/bin/env python3
"""
Square-Root Dual Unscented Kalman Filter (SR-DUKF) for joint estimation of
State of Charge (SoC) and battery parameters.

This implementation replaces the EKF with a UKF for improved accuracy, especially
in the presence of non-linearities. It also uses a square-root formulation for
the covariance matrices to ensure numerical stability and prevent divergence.
"""

import numpy as np
from scipy.linalg import cholesky, qr

class DualUKF:
    def __init__(self, dt, C_nom, R0_nom, ocv_lstm_model, alpha=1e-3, beta=2., kappa=0.):
        self.dt = dt
        self.ocv_lstm_model = ocv_lstm_model

        # --- Filter Tuning Parameters ---
        # Process noise: Lower values indicate more trust in the model's prediction.
        self.q_state = np.diag([1e-7, 1e-7]) 
        self.q_param = np.diag([1e-8, 1e-9])
        # Measurement noise: Higher values indicate less trust in the sensor measurements.
        self.r_state = np.array([[1e-1]]) # Increased from 1e-2
        self.r_param = np.array([[1e-1]]) # Increased from 1e-2

        # --- State Filter (UKF) Initialization ---
        self.x = np.array([0.9, 0.0])  # State: [SoC, U_d]
        self.n_x = len(self.x)
        self.S_x = cholesky(np.diag([1e-4, 1e-4])) # Cholesky factor of state covariance

        # --- Parameter Filter (UKF) Initialization ---
        self.theta = np.array([C_nom, R0_nom]) # Parameters: [Q_max, R_0]
        self.n_theta = len(self.theta)
        self.S_theta = cholesky(np.diag([1e-2, 1e-4])) # Cholesky factor of param covariance

        # --- UKF Sigma Point Generation Parameters ---
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_x = alpha**2 * (self.n_x + kappa) - self.n_x
        self.lambda_theta = alpha**2 * (self.n_theta + kappa) - self.n_theta
        self.W_m_x, self.W_c_x = self._compute_weights(self.n_x, self.lambda_x)
        self.W_m_theta, self.W_c_theta = self._compute_weights(self.n_theta, self.lambda_theta)

        # --- Other fixed parameters ---
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
        soc_next = soc_k - (self.eta * i_k / (Q_max * 3600)) * self.dt
        u_d_next = u_d_k * np.exp(-self.dt / (self.R_D * self.C_D)) + self.R_D * (1 - np.exp(-self.dt / (self.R_D * self.C_D))) * i_k
        return np.array([soc_next, u_d_next])

    def _observation(self, x, theta, temp_k, i_k):
        soc_k, u_d_k = x
        R_0 = theta[1]
        model_input = np.array([[soc_k, temp_k]]).reshape((1, 1, 2))
        ocv = self.ocv_lstm_model.predict(model_input, verbose=0)[0, 0]
        return ocv - i_k * R_0 - u_d_k

    def step(self, v_meas, i_k, temp_k):
        # --- State Estimation ---
        # Generate sigma points for state
        sigma_x = self._generate_sigma_points(self.x, self.S_x, self.lambda_x)
        
        # Propagate sigma points through state transition model
        sigma_x_pred = np.array([self._state_transition(s, i_k) for s in sigma_x])
        
        # Calculate predicted mean and covariance (Cholesky factor)
        x_pred = np.dot(self.W_m_x, sigma_x_pred)
        X = sigma_x_pred - x_pred[np.newaxis, :]
        # Numerically stable way to update Cholesky factor of covariance
        R = qr(np.sqrt(self.W_c_x[1]) * X[1:, :].T)[0].T
        S_x_pred = cholesky(R @ R.T + self.q_state, lower=False)

        # Propagate predicted sigma points through observation model
        sigma_y_pred = np.array([self._observation(s, self.theta, temp_k, i_k) for s in sigma_x_pred])
        y_pred = np.dot(self.W_m_x, sigma_y_pred)
        
        # Calculate innovation covariance and cross-covariance
        Y = (sigma_y_pred - y_pred)[:, np.newaxis]
        P_yy = (Y.T @ np.diag(self.W_c_x) @ Y) + self.r_state
        P_xy = X.T @ np.diag(self.W_c_x) @ Y
        
        # Kalman gain (using simple division for scalar innovation)
        K = P_xy / P_yy
        
        # Update state and covariance
        self.x = x_pred + K.flatten() * (v_meas - y_pred)
        U = K @ cholesky(P_yy, lower=False)
        self.S_x = cholesky(S_x_pred.T @ S_x_pred - U @ U.T, lower=False)
        self.x[0] = np.clip(self.x[0], 0, 1) # Enforce SoC bounds

        # --- Parameter Estimation (runs periodically) ---
        if self.step_counter % self.parameter_filter_update_rate == 0:
            # Generate sigma points for parameters
            sigma_theta = self._generate_sigma_points(self.theta, self.S_theta, self.lambda_theta)
            
            # Parameter transition is random walk, so sigma_theta_pred = sigma_theta
            theta_pred = np.dot(self.W_m_theta, sigma_theta)
            Theta = sigma_theta - theta_pred[np.newaxis, :]
            R_theta = qr(np.sqrt(self.W_c_theta[1]) * Theta[1:, :].T)[0].T
            S_theta_pred = cholesky(R_theta @ R_theta.T + self.q_param, lower=False)

            # Propagate through observation model
            sigma_y_theta = np.array([self._observation(self.x, s, temp_k, i_k) for s in sigma_theta])
            y_theta_pred = np.dot(self.W_m_theta, sigma_y_theta)
            
            Y_theta = (sigma_y_theta - y_theta_pred)[:, np.newaxis]
            P_yy_theta = (Y_theta.T @ np.diag(self.W_c_theta) @ Y_theta) + self.r_param
            P_theta_y = Theta.T @ np.diag(self.W_c_theta) @ Y_theta
            
            K_theta = P_theta_y / P_yy_theta
            
            self.theta = theta_pred + K_theta.flatten() * (v_meas - y_theta_pred)
            U_theta = K_theta @ cholesky(P_yy_theta, lower=False)
            self.S_theta = cholesky(S_theta_pred.T @ S_theta_pred - U_theta @ U_theta.T, lower=False)

            # Defensive checks for parameters
            if self.theta[0] < 0: # Capacity
                print(f"Warning: Negative capacity estimated ({self.theta[0]:.2f}). Resetting.")
                self.theta[0] = 1e-2
            if self.theta[1] < 0: # Resistance
                print(f"Warning: Negative resistance estimated ({self.theta[1]:.2f}). Resetting.")
                self.theta[1] = 1e-4

        self.step_counter += 1
        return self.x, self.theta
