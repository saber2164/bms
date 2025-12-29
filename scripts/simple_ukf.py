#!/usr/bin/env python3
"""
Simplified DualUKF with LINEAR OCV model.
No neural network - uses simple linear relationship: OCV = a*SoC + b
"""

import numpy as np
from scipy.linalg import cholesky, qr, LinAlgError

class SimplifiedDualUKF:
    """Dual UKF with linear OCV model - guaranteed monotonic and physically sound"""
    
    def __init__(self, dt, C_nom, R0_nom, ocv_a=1.2, ocv_b=3.0, alpha=0.1, beta=2., kappa=0.):
        """
        Args:
            dt: Time step (seconds)
            C_nom: Nominal capacity (Ah)
            R0_nom: Nominal internal resistance (Ohms)
            ocv_a: OCV slope (V per unit SoC), default 1.2
            ocv_b: OCV offset (V at SoC=0), default 3.0
                   This gives: 3.0V at 0% SoC, 4.2V at 100% SoC
        """
        self.dt = dt
        self.ocv_a = ocv_a
        self.ocv_b = ocv_b
        
        # RC pair parameters (for voltage dynamics)
        self.R_D = 0.01  # Polarization resistance
        self.C_D = 2000  # Polarization capacitance
        self.eta = 1.0   # Coulombic efficiency
        
        # Filter tuning - Increased noise for faster convergence
        self.q_state = np.diag([1e-5, 1e-5])  # Increased from 1e-6
        self.q_param = np.diag([1e-6, 1e-7])  # Increased from 1e-8
        self.r_state = np.array([[5e-3]])     # Updated from 1e-2
        self.r_param = np.array([[1e-2]])
        
        # State filter initialization
        self.x = np.array([0.9, 0.0])  # [SoC, U_d]
        self.n_x = len(self.x)
        self.S_x = self._robust_cholesky(np.diag([1e-2, 1e-2])) # Increased from 1e-4
        
        # Parameter filter initialization
        self.theta = np.array([C_nom, R0_nom])  # [Q_max, R_0]
        self.n_theta = len(self.theta)
        self.S_theta = self._robust_cholesky(np.diag([1e-2, 1e-4])) # Increased from 1e-4
        
        # UKF sigma point parameters
        self.lambda_x = alpha**2 * (self.n_x + kappa) - self.n_x
        self.lambda_theta = alpha**2 * (self.n_theta + kappa) - self.n_theta
        
        # Weights
        self.W_m_x = np.concatenate([[self.lambda_x / (self.n_x + self.lambda_x)],
                                      0.5 / (self.n_x + self.lambda_x) * np.ones(2*self.n_x)])
        self.W_c_x = np.copy(self.W_m_x)
        self.W_c_x[0] += (1 - alpha**2 + beta)
        
        self.W_m_theta = np.concatenate([[self.lambda_theta / (self.n_theta + self.lambda_theta)],
                                          0.5 / (self.n_theta + self.lambda_theta) * np.ones(2*self.n_theta)])
        self.W_c_theta = np.copy(self.W_m_theta)
        self.W_c_theta[0] += (1 - alpha**2 + beta)
        
        self.step_counter = 0
        self.parameter_filter_update_rate = 100 # Updated from 10
    
    def _robust_cholesky(self, M):
        """
        Robust Cholesky decomposition with jitter and symmetrization.
        Ensures positive definiteness.
        """
        try:
            # Symmetrize
            M = (M + M.T) / 2.0
            # Add jitter
            M = M + 1e-12 * np.eye(M.shape[0])
            return cholesky(M, lower=False)
        except LinAlgError:
            # Fallback: add more jitter if it fails
            try:
                M = M + 1e-9 * np.eye(M.shape[0])
                return cholesky(M, lower=False)
            except LinAlgError:
                # Last resort: diagonalize (extreme fallback)
                return np.sqrt(np.diag(np.diag(M)))

    def _linear_ocv(self, soc):
        """Linear OCV model - guaranteed monotonic"""
        return self.ocv_a * soc + self.ocv_b
    
    def _generate_sigma_points(self, mean, S, lambda_val):
        """Generate sigma points"""
        n = len(mean)
        sigma = np.zeros((2*n + 1, n))
        sigma[0] = mean
        sqrt_term = np.sqrt(n + lambda_val) * S.T
        sigma[1:n+1] = mean + sqrt_term
        sigma[n+1:] = mean - sqrt_term
        return sigma
    
    def _state_transition(self, x, i_k):
        """State transition: [SoC, U_d]"""
        soc_k, u_d_k = x
        Q_max = self.theta[0]
        
        # Coulomb counting
        soc_next = soc_k - (self.eta * i_k / (Q_max * 3600)) * self.dt
        
        # RC dynamics
        exp_val = np.exp(-self.dt / (self.R_D * self.C_D))
        u_d_next = u_d_k * exp_val + self.R_D * (1 - exp_val) * i_k
        
        return np.array([soc_next, u_d_next])
    
    def _batch_observation(self, sigma_points_x, theta, temp_k, i_k):
        """
        Vectorized observation - uses LINEAR OCV model
        V_terminal = OCV(SoC) - I*R0 - U_d
        """
        R_0 = theta[1]
        soc_values = sigma_points_x[:, 0]
        u_d_values = sigma_points_x[:, 1]
        
        # Linear OCV - simple and monotonic!
        ocv_preds = self.ocv_a * soc_values + self.ocv_b
        
        # Terminal voltage
        v_preds = ocv_preds - i_k * R_0 - u_d_values
        
        return v_preds
    
    def step(self, v_meas, i_k, temp_k):
        """Single UKF step"""
        # State estimation
        sigma_x = self._generate_sigma_points(self.x, self.S_x, self.lambda_x)
        sigma_x_pred = np.array([self._state_transition(s, i_k) for s in sigma_x])
        x_pred = np.dot(self.W_m_x, sigma_x_pred)
        
        X = sigma_x_pred - x_pred[np.newaxis, :]
        R = qr(np.sqrt(self.W_c_x[1]) * X[1:, :].T)[0].T
        S_x_pred = self._robust_cholesky(R @ R.T + self.q_state)
        
        # Measurement prediction
        sigma_y_pred = self._batch_observation(sigma_x_pred, self.theta, temp_k, i_k)
        y_pred = np.dot(self.W_m_x, sigma_y_pred)
        
        Y = (sigma_y_pred - y_pred)[:, np.newaxis]
        P_yy = (Y.T @ np.diag(self.W_c_x) @ Y) + self.r_state
        P_xy = X.T @ np.diag(self.W_c_x) @ Y
        
        K = P_xy / P_yy
        
        # Update
        self.x = x_pred + K.flatten() * (v_meas - y_pred)
        U = K @ self._robust_cholesky(P_yy)
        self.S_x = self._robust_cholesky(S_x_pred.T @ S_x_pred - U @ U.T)
        self.x[0] = np.clip(self.x[0], 0, 1)
        5
        # Parameter estimation (periodic)
        if self.step_counter % self.parameter_filter_update_rate == 0:
            sigma_theta = self._generate_sigma_points(self.theta, self.S_theta, self.lambda_theta)
            theta_pred = np.dot(self.W_m_theta, sigma_theta)
            Theta = sigma_theta - theta_pred[np.newaxis, :]
            R_theta = qr(np.sqrt(self.W_c_theta[1]) * Theta[1:, :].T)[0].T
            S_theta_pred = self._robust_cholesky(R_theta @ R_theta.T + self.q_param)
            
            sigma_y_theta = np.array([self._batch_observation(sigma_x_pred, th, temp_k, i_k).mean() 
                                      for th in sigma_theta])
            y_pred_theta = np.dot(self.W_m_theta, sigma_y_theta)
            
            Y_theta = (sigma_y_theta - y_pred_theta)[:, np.newaxis]
            P_yy_theta = (Y_theta.T @ np.diag(self.W_c_theta) @ Y_theta) + self.r_param
            P_theta_y = Theta.T @ np.diag(self.W_c_theta) @ Y_theta
            
            K_theta = P_theta_y / P_yy_theta
            
            # Store previous Q for smoothing
            prev_Q = self.theta[0]
            
            self.theta = theta_pred + K_theta.flatten() * (v_meas - y_pred_theta)
            
            # Apply exponential smoothing to Q_max
            self.theta[0] = 0.95 * self.theta[0] + 0.05 * prev_Q
            
            U_theta = K_theta @ self._robust_cholesky(P_yy_theta)
            self.S_theta = self._robust_cholesky(S_theta_pred.T @ S_theta_pred - U_theta @ U_theta.T)
            
            # Constrain parameters
            self.theta[0] = np.clip(self.theta[0], 0.5, 5.0)  # Q_max
            self.theta[1] = np.clip(self.theta[1], 0.001, 0.5)  # R_0
        
        self.step_counter += 1
        return self.x[0], self.theta[0], self.theta[1]
