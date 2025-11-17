#!/usr/bin/env python3
"""
Dual Extended Kalman Filter (DEKF) for joint estimation of State of Charge (SoC)
and battery parameters (Q_max, R_0).

This implementation is based on the principles outlined in various academic papers
on joint state and parameter estimation for lithium-ion batteries. The core idea
is to run two EKFs in parallel:
- A State EKF for estimating the fast-changing SoC.
- A Parameter EKF for estimating slow-changing parameters like capacity and resistance.

The filters are coupled, allowing the model to adapt to battery degradation (SoH changes).
"""

import numpy as np

class DualEKF:
    def __init__(self, dt, C_nom, R0_nom, R_D_nom, C_D_nom, eta_nom, ocv_lstm_model):
        """
        Initializes the Dual Extended Kalman Filter.

        Args:
            dt (float): Time step in seconds.
            C_nom (float): Initial nominal capacity (Ah).
            R0_nom (float): Initial internal resistance (Ohms).
            R_D_nom (float): Initial diffusion resistance (Ohms).
            C_D_nom (float): Initial diffusion capacitance (Farads).
            eta_nom (float): Initial coulombic efficiency.
            ocv_lstm_model: A trained Keras model to predict OCV from SoC and Temperature.
        """
        self.dt = dt
        self.ocv_lstm_model = ocv_lstm_model

        # --- State Filter Initialization ---
        # State vector: x = [SoC, U_d] (State of Charge, Diffusion Voltage)
        self.x_k = np.array([0.9, 0.0])  # Initial state (e.g., 90% SoC)
        # State covariance matrix P
        self.P_k = np.diag([1e-4, 1e-4])
        # Process noise covariance Q
        self.Q_k = np.diag([1e-6, 1e-6])
        # Measurement noise covariance R
        self.R_k = np.array([1e-3])

        # --- Parameter Filter Initialization ---
        # Parameter vector: theta = [Q_max, R_0]
        self.theta_k = np.array([C_nom, R0_nom]) # Initial parameters
        # Parameter covariance matrix P_theta
        self.p_theta_k = np.diag([1e-2, 1e-4])
        # Parameter process noise covariance Q_theta
        self.Q_theta_k = np.diag([1e-7, 1e-8]) # Small noise to allow for slow drift
        # Parameter measurement noise covariance R_theta
        self.R_theta_k = np.array([1e-3])

        # Store other parameters that are not estimated by the DEKF for now
        self.R_D = R_D_nom
        self.C_D = C_D_nom
        self.eta = eta_nom

        # Update frequency for the parameter filter
        self.parameter_filter_update_rate = 100 # Update every 100 steps
        self.gradient_update_rate = 10 # Update gradient every 10 steps
        self.step_counter = 0
        self.ocv_grad = 0 # Initial gradient

    def _state_transition_model(self, x, i_k, Q_max):
        """ The state transition function f(x, u) for the State EKF. """
        soc_k, u_d_k = x
        soc_next = soc_k - (self.eta * i_k / (Q_max * 3600)) * self.dt
        u_d_next = u_d_k * np.exp(-self.dt / (self.R_D * self.C_D)) + self.R_D * (1 - np.exp(-self.dt / (self.R_D * self.C_D))) * i_k
        return np.array([soc_next, u_d_next])

    def _observation_model(self, x, i_k, R_0, temp_k):
        """ The observation function h(x, u) for both EKFs. """
        soc_k, u_d_k = x
        # Reshape for LSTM: (1, 1, 2) -> (batch_size, timesteps, features)
        model_input = np.array([[soc_k, temp_k]]).reshape((1, 1, 2))
        # Use the LSTM to get the OCV
        ocv = self.ocv_lstm_model.predict(model_input, verbose=0)[0, 0]
        v_terminal = ocv - i_k * R_0 - u_d_k
        return v_terminal

    def predict_state(self, i_k):
        """ Prediction step for the State EKF. """
        Q_max = self.theta_k[0] # Use current capacity estimate

        # Jacobian of the state transition model
        A_k = np.array([
            [1, 0],
            [0, np.exp(-self.dt / (self.R_D * self.C_D))]
        ])

        # Predict state and covariance
        self.x_k = self._state_transition_model(self.x_k, i_k, Q_max)
        self.P_k = A_k @ self.P_k @ A_k.T + self.Q_k

    def update_state(self, v_meas, i_k, temp_k):
        """ Update step for the State EKF. """
        R_0 = self.theta_k[1] # Use current resistance estimate

        # Update OCV gradient periodically
        if self.step_counter % self.gradient_update_rate == 0:
            eps = 1e-5
            soc_plus_eps = self.x_k[0] + eps
            
            input_at_x = np.array([[self.x_k[0], temp_k]]).reshape((1, 1, 2))
            input_at_x_plus_eps = np.array([[soc_plus_eps, temp_k]]).reshape((1, 1, 2))

            ocv_at_x = self.ocv_lstm_model.predict(input_at_x, verbose=0)[0, 0]
            ocv_at_x_plus_eps = self.ocv_lstm_model.predict(input_at_x_plus_eps, verbose=0)[0, 0]
            
            self.ocv_grad = (ocv_at_x_plus_eps - ocv_at_x) / eps

        C_k = np.array([[self.ocv_grad, -1]])

        # Kalman Gain
        K_k = self.P_k @ C_k.T @ np.linalg.inv(C_k @ self.P_k @ C_k.T + self.R_k)

        # Update state and covariance
        innovation = v_meas - self._observation_model(self.x_k, i_k, R_0, temp_k)
        self.x_k = self.x_k + K_k.flatten() * innovation
        self.P_k = (np.eye(2) - K_k @ C_k) @ self.P_k

        # --- Constraint to keep SoC within [0, 1] ---
        self.x_k[0] = np.clip(self.x_k[0], 0, 1)

    def predict_parameters(self):
        """ Prediction step for the Parameter EKF. """
        # Parameters are assumed to be a random walk
        # theta_k = theta_{k-1}
        # Jacobian is identity
        A_theta_k = np.eye(2)
        self.p_theta_k = A_theta_k @ self.p_theta_k @ A_theta_k.T + self.Q_theta_k

    def update_parameters(self, v_meas, i_k, temp_k):
        """ Update step for the Parameter EKF. """
        soc_k = self.x_k[0] # Use current SoC estimate

        # Jacobian of the observation model with respect to the parameters [Q_max, R_0]
        # C_theta_k = [dh/dQ_max, dh/dR_0]
        # The dependency of OCV on Q_max is implicit through the SoC, making the Jacobian complex.
        # A common simplification is to assume dh/dQ_max is negligible for a single update step
        # and focus on the direct impact of R_0. This enhances stability.
        # For this implementation, we only update R_0 and Q_max is treated as a random walk.
        C_theta_k = np.array([[0, -i_k]])

        # Kalman Gain
        K_theta_k = self.p_theta_k @ C_theta_k.T @ np.linalg.inv(C_theta_k @ self.p_theta_k @ C_theta_k.T + self.R_theta_k)

        # Update parameters and covariance
        innovation = v_meas - self._observation_model(self.x_k, i_k, self.theta_k[1], temp_k)
        self.theta_k = self.theta_k + K_theta_k.flatten() * innovation
        self.p_theta_k = (np.eye(2) - K_theta_k @ C_theta_k) @ self.p_theta_k

        # --- Constraint to prevent negative resistance ---
        if self.theta_k[1] < 0:
            self.theta_k[1] = 1e-4 # Reset to a small positive value

    def step(self, v_meas, i_k, temp_k):
        """ Perform one full step of the DEKF. """
        # --- State Filter (runs every step) ---
        self.predict_state(i_k)
        self.update_state(v_meas, i_k, temp_k)

        # --- Parameter Filter (runs periodically) ---
        if self.step_counter % self.parameter_filter_update_rate == 0:
            self.predict_parameters()
            self.update_parameters(v_meas, i_k, temp_k)

        self.step_counter += 1
        return self.x_k, self.theta_k
