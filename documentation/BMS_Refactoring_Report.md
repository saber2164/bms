# Project Report: Refactoring of a Battery Management System's State Estimation Model

## Section 1: Overview of the Project

### 1.1 Initial Problem Statement
The project began with a hybrid Extended Kalman Filter (EKF) and LSTM model for battery State of Charge (SoC) estimation. This initial model was found to be inaccurate and unstable, with its estimates failing to generalize and drifting significantly as the battery aged (SoH degradation) and when operating under different temperatures.

### 1.2 Project Goal
The primary goal was to refactor the Python codebase to create a robust, adaptive, and state-of-the-art hybrid model for battery state estimation. The new architecture needed to solve the parameter drift problem and improve the stability and accuracy of the SoC estimates.

### 1.3 Summary of the Final Implemented Solution
The final implementation is a sophisticated and robust state estimation system based on a **Square-Root Dual Unscented Kalman Filter (SR-DUKF)**. This filter is coupled with a Long Short-Term Memory (LSTM) neural network that provides a data-driven model of the battery's Open-Circuit Voltage (OCV). This architecture is designed to be adaptive, numerically stable, and accurate in the face of the battery's complex non-linear behavior.

---

## Section 2: Summary of Tasks and Implementations

The project progressed through several stages of refactoring and debugging:

1.  **Initial Architecture (EKF+LSTM):** The starting point was a simple EKF that used an LSTM to predict State of Health (SoH) and Remaining Useful Life (RUL). This approach was found to be inaccurate because the EKF's underlying model did not adapt to changes in the battery's parameters as it aged.

2.  **Refactoring to Dual EKF (DEKF):**
    *   **Approach:** To address the parameter drift, the architecture was first refactored to a Dual EKF (DEKF). This involved creating two communicating EKFs: a state filter for the fast-changing SoC and a parameter filter to estimate the slow-changing capacity (`Q_max`) and internal resistance (`R_0`).
    *   **Implementation:** The `scripts/dekf_soc.py` script was created to house the `DualEKF` class.

3.  **LSTM for OCV Estimation:**
    *   **Approach:** The role of the LSTM was changed to model the highly non-linear relationship between SoC, temperature, and Open-Circuit Voltage (OCV). This data-driven OCV model provides a much more accurate observation model for the Kalman filter.
    *   **Implementation:** The `scripts/train_dekf_lstm.py` script was created to train the OCV-LSTM model. This script also implements a simulated transfer learning workflow to allow for pre-training on a large dataset and fine-tuning on the project's specific data.

4.  **Refactoring to Square-Root Dual UKF (SR-DUKF):**
    *   **Approach:** During testing, the DEKF was found to be unstable and prone to divergence, a common issue with EKFs in highly non-linear systems. To solve this, the entire filter was refactored to a Square-Root Dual Unscented Kalman Filter (SR-DUKF).
    *   **Implementation:** The `scripts/ukf_soc.py` script was created, containing the `DualUKF` class. This implementation is significantly more robust and numerically stable.

5.  **Validation and Debugging:**
    *   **OCV Validation:** The `scripts/validate_ocv.py` script was created to visualize the learned OCV-SoC curve from the LSTM model, allowing for crucial inspection of the model's behavior.
    *   **Batch Testing:** The `scripts/get_good_files.py` and `scripts/batch_infer.py` scripts were created to automate the process of testing the filter on a large number of files and detecting anomalies.
    *   **Iterative Debugging:** The development process involved extensive debugging of the filters, including fixing issues related to negative resistance, negative SoC, and numerical instability (singular matrices).

---

## Section 3: Comparison of Approaches (EKF vs. UKF)

### 3.1 Extended Kalman Filter (EKF)
*   **How it Works:** The EKF handles non-linearity by linearizing the system model at every time step using Jacobians (partial derivatives). It then applies the standard linear Kalman filter equations to this linearized model.
*   **Pros:** It is computationally less expensive than the UKF.
*   **Cons:** The linearization can be a poor approximation for highly non-linear systems, such as the OCV curve of a battery. This inaccuracy can lead to poor estimates and filter divergence.

### 3.2 Unscented Kalman Filter (UKF)
*   **How it Works:** The UKF uses a more sophisticated approach called the Unscented Transform. Instead of linearizing the model, it uses a set of carefully chosen sample points (called "sigma points") that capture the mean and covariance of the state distribution. These sigma points are then propagated through the *true* non-linear model, and a new mean and covariance are calculated from the propagated points.
*   **Pros:** It is significantly more accurate than the EKF for non-linear systems and does not require the calculation of Jacobians.
*   **Cons:** It is computationally more expensive due to the need to propagate multiple sigma points.

### 3.3 Square-Root UKF (Our Implementation)
*   **How it Works:** This is an advanced variant of the UKF that never explicitly calculates the full covariance matrix. Instead, it propagates the Cholesky factor (or "square root") of the covariance matrix.
*   **Pros:** It is numerically more stable and robust. It guarantees that the covariance matrix remains positive semi-definite, which is a critical property that can be lost in standard UKF/EKF implementations due to numerical round-off errors, leading to filter divergence.

---

## Section 4: The Math and Science Behind the Methods

### 4.1 Kalman Filtering
A Kalman filter is an optimal estimation algorithm for linear systems with Gaussian noise. It operates in a two-step, predict-update cycle:
1.  **Predict:** The filter uses the system's model to predict the next state and its uncertainty.
2.  **Update:** The filter uses a measurement (from a sensor) to correct the prediction. The amount of correction is determined by the Kalman Gain, which balances the uncertainty of the prediction with the uncertainty of the measurement.

### 4.2 The Battery Model (Equivalent Circuit)
The filter's internal model is based on a common equivalent circuit for a lithium-ion battery. This model includes:
*   A voltage source representing the Open-Circuit Voltage (OCV).
*   A resistor (`R_0`) representing the internal resistance.
*   One or more RC pairs (`R_D`, `C_D`) representing the battery's diffusion dynamics.

The **state equations** describe how the SoC and the voltage across the RC pair evolve over time. The **observation equation** relates these internal states to the measurable terminal voltage.

### 4.3 The Unscented Transform
The Unscented Transform is the core of the UKF. It is a method for calculating the statistics of a random variable that undergoes a non-linear transformation. It is based on the principle that it is easier to approximate a probability distribution than it is to approximate a non-linear function. It works by:
1.  Generating a small, fixed number of "sigma points" from the state distribution.
2.  Propagating these points through the true non-linear function.
3.  Calculating the mean and covariance of the transformed points.

This method provides a more accurate estimation of the output distribution's mean and covariance than the linearization used by the EKF.

### 4.4 Dual Filtering for Joint Estimation
The Dual Filter architecture is used to simultaneously estimate the battery's internal state (which changes quickly) and its parameters (which change slowly as the battery ages).
*   **State Filter (UKF):** Runs at a fast timescale (e.g., every second) to estimate the SoC. It uses the latest parameter estimates from the parameter filter.
*   **Parameter Filter (UKF):** Runs at a slower timescale (e.g., every 100 seconds) to estimate `Q_max` and `R_0`. It uses the latest SoC estimate from the state filter to inform its estimation.

This coupling allows the model to adapt to the battery's changing characteristics over its lifetime.

---

## Section 5: Conclusion and Next Steps

The project has successfully refactored the battery state estimation model from a simple EKF-based approach to a state-of-the-art Square-Root Dual Unscented Kalman Filter (SR-DUKF) architecture. The new implementation is more robust, numerically stable, and better equipped to handle the non-linearities of a real-world battery system.

The recommended next steps are:
1.  **Train the OCV-LSTM Model on More Data:** The accuracy of the entire system is highly dependent on the OCV model. This model should be retrained using all of the available valid data files (we found 5,609 of them).
2.  **Systematic Filter Tuning:** The noise parameters (`Q` and `R`) in the `ukf_soc.py` script should be carefully tuned to optimize the filter's performance. This is typically an iterative process that involves comparing the filter's estimates to a known ground truth.
