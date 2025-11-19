
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Input

class RULPredictor:
    """
    A class to predict Remaining Useful Life (RUL) of a battery using a hybrid CNN-LSTM model.

    This class takes a sequence of historical State of Health (SoH) values,
    trains a model to predict future SoH, and estimates the RUL based on when
    the SoH is predicted to reach a defined end-of-life threshold (e.g., 80%).
    """

    def __init__(self, sequence_length: int = 10, end_of_life_soh: float = 0.8, n_features: int = 1, hp=None):
        """
        Initializes the RULPredictor.

        Args:
            sequence_length (int): The number of historical SoH values to use as input (N cycles).
            end_of_life_soh (float): The SoH threshold that defines the battery's end-of-life.
            n_features (int): The number of input features per time step.
            hp (keras_tuner.HyperParameters): Optional HyperParameters object for tuning.
        """
        if not (0 < end_of_life_soh < 1):
            raise ValueError("end_of_life_soh must be a float between 0 and 1.")

        self.sequence_length = sequence_length
        self.end_of_life_soh = end_of_life_soh
        self.n_features = n_features
        self.hp = hp
        self.model = self.build_model()

    def build_model(self, hp=None) -> Sequential:
        """
        Builds the hybrid CNN-LSTM model architecture for direct RUL prediction.

        Returns:
            A compiled TensorFlow/Keras Sequential model.
        """
        # Hyperparameters
        if self.hp:
            conv_filters = self.hp.Int('conv_filters', min_value=16, max_value=64, step=16)
            conv_kernel = self.hp.Int('conv_kernel', min_value=3, max_value=7, step=2)
            lstm_units = self.hp.Int('lstm_units', min_value=32, max_value=128, step=32)
            dense_units = self.hp.Int('dense_units', min_value=16, max_value=64, step=16)
            learning_rate = self.hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        else:
            conv_filters = 16
            conv_kernel = 3
            lstm_units = 32
            dense_units = 16
            learning_rate = 0.001

        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features), name="soh_sequence_input"),
            Conv1D(filters=conv_filters, kernel_size=conv_kernel, activation='relu', padding='causal'),
            LSTM(units=lstm_units, activation='tanh', return_sequences=False),
            Dense(units=dense_units, activation='relu'),
            Dense(units=1, name="rul_output")  # Directly predict RUL
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        return model

    def predict_rul(self, soh_sequence: np.ndarray) -> float:
        """
        Predicts the Remaining Useful Life (RUL) for a battery given its
        recent SoH history.

        Args:
            soh_sequence (np.ndarray): A numpy array of recent SoH values.

        Returns:
            The predicted RUL in cycles.
        """
        if len(soh_sequence) != self.sequence_length:
            # Pad or truncate if necessary, though ideally the input should be correct
            soh_sequence = np.array(list(soh_sequence)).reshape(1, self.sequence_length, 1)
        
        rul = self.model.predict(soh_sequence, verbose=0)[0][0]
        return rul

    def _create_sequences(self, data: np.ndarray, rul_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates input sequences and corresponding RUL targets.

        Args:
            data (np.ndarray): The historical SoH data.
            rul_data (np.ndarray): The corresponding RUL for each point in `data`.

        Returns:
            A tuple containing:
            - X (np.ndarray): Input sequences of shape (n_samples, sequence_length, 1).
            - y (np.ndarray): Target RUL values.
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(rul_data[i + self.sequence_length])
        return np.array(X), np.array(y)

# Example Usage (for demonstration purposes)
if __name__ == '__main__':
    # --- 1. Generate synthetic battery degradation data ---
    # Let's simulate a battery that starts at 100% SoH and degrades over cycles.
    # SoH(c) = 1.0 - a*c - b*c^2
    total_cycles = 1500
    cycles = np.arange(total_cycles)
    a = 5e-5  # Linear degradation
    b = 1e-7  # Quadratic degradation
    noise = np.random.normal(0, 0.002, total_cycles)
    soh_true = 1.0 - a * cycles - b * cycles**2 + noise
    # Ensure SoH doesn't go above 1.0
    soh_true = np.clip(soh_true, None, 1.0)

    # --- 2. Instantiate and Train the RUL Predictor ---
    # We will train the model on the first 60% of the battery's life.
    train_split_index = int(total_cycles * 0.6)
    soh_train_data = soh_true[:train_split_index]

    sequence_len = 50  # Use last 50 cycles to predict the next one
    rul_predictor = RULPredictor(sequence_length=sequence_len, end_of_life_soh=0.8)
    
    print("--- Training RUL Predictor ---")
    rul_predictor.train(soh_train_data, epochs=50, batch_size=16, verbose=1)
    print("\n--- Training Complete ---")

    # --- 3. Predict RUL at a specific point in the battery's life ---
    # Let's test the model at a later point, e.g., at cycle 1000
    current_cycle_test = 1000
    start_index = current_cycle_test - sequence_len
    end_index = current_cycle_test
    
    if start_index < 0:
        print(f"Not enough historical data at cycle {current_cycle_test} to make a prediction.")
    else:
        # Get the sequence of SoH values leading up to the test cycle
        recent_soh = soh_true[start_index:end_index]

        # Predict the RUL
        predicted_rul = rul_predictor.predict_rul(current_cycle_test, recent_soh)

        # --- 4. Calculate the actual RUL for comparison ---
        try:
            eol_actual_cycle = np.where(soh_true < 0.8)[0][0]
            actual_rul = eol_actual_cycle - current_cycle_test
            print(f"\n--- RUL Prediction at Cycle {current_cycle_test} ---")
            print(f"Input SoH Sequence (last 5 values): {np.round(recent_soh[-5:], 4)}")
            print(f"Predicted RUL: {predicted_rul:.2f} cycles")
            print(f"Actual RUL: {actual_rul:.2f} cycles")
        except IndexError:
            print("Battery has not yet reached End-of-Life in the true data.")

    # --- 5. Visualize the prediction (Optional) ---
    import matplotlib.pyplot as plt

    # Generate a full predicted trajectory from the test point
    test_sequence = soh_true[start_index:end_index]
    predicted_trajectory = rul_predictor.predict_soh_trajectory(test_sequence)
    
    # Create cycle axis for the predicted part
    predicted_cycles = np.arange(current_cycle_test, current_cycle_test + len(predicted_trajectory) - sequence_len)

    plt.figure(figsize=(12, 7))
    plt.plot(cycles, soh_true, label='Actual SoH Trajectory', color='blue', alpha=0.7)
    plt.plot(predicted_cycles, predicted_trajectory[sequence_len-1:], label='Predicted SoH Trajectory', color='red', linestyle='--')
    plt.axhline(y=0.8, color='k', linestyle='--', label='End-of-Life (80% SoH)')
    plt.axvline(x=current_cycle_test, color='green', linestyle=':', label=f'Prediction Point (Cycle {current_cycle_test})')
    
    if predicted_rul is not None:
        predicted_eol = current_cycle_test + predicted_rul
        plt.axvline(x=predicted_eol, color='magenta', linestyle=':', label=f'Predicted EOL ({predicted_eol:.0f} cycles)')

    plt.title('Battery SoH Degradation and RUL Prediction')
    plt.xlabel('Cycle Number')
    plt.ylabel('State of Health (SoH)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.75, 1.02) # Zoom in on the relevant SoH range
    plt.xlim(0, total_cycles)
    # The plot will not be shown in this environment, but this demonstrates how to visualize it.
    # To display, you would typically call plt.show()
    # To save, you would call plt.savefig('rul_prediction_plot.png')
    print("\nPlot generation code is ready (matplotlib required).")
