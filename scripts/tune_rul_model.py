import os
import tensorflow as tf
import keras_tuner as kt
from train_rul_model import get_train_test_data
from rul_predictor import RULPredictor

def build_model_wrapper(sequence_length, end_of_life_soh, n_features):
    """
    Wrapper to pass additional arguments to the model building function.
    """
    def build_model(hp):
        predictor = RULPredictor(
            sequence_length=sequence_length,
            end_of_life_soh=end_of_life_soh,
            n_features=n_features,
            hp=hp
        )
        return predictor.model
    return build_model

def main():
    # Configuration
    SOH_DATA_FILE = 'soh_features.csv'
    METADATA_FILE = 'cleaned_dataset/metadata.csv'
    SEQUENCE_LENGTH = 15
    END_OF_LIFE_SOH = 0.8
    MAX_TRIALS = 5
    EXECUTIONS_PER_TRIAL = 1
    EPOCHS = 50
    BATCH_SIZE = 64
    TUNER_DIR = 'outputs/tuning'
    PROJECT_NAME = 'rul_optimization'

    # Load Data
    print("Loading data for tuning...")
    X_train, y_train, X_test, y_test, n_features = get_train_test_data(
        SOH_DATA_FILE, METADATA_FILE, SEQUENCE_LENGTH, END_OF_LIFE_SOH, test_split=0.2
    )

    if X_train is None:
        print("Failed to load data.")
        return

    # Initialize Tuner
    print("\n--- Initializing Keras Tuner ---")
    tuner = kt.RandomSearch(
        build_model_wrapper(SEQUENCE_LENGTH, END_OF_LIFE_SOH, n_features),
        objective='val_mean_absolute_error',
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        directory=TUNER_DIR,
        project_name=PROJECT_NAME,
        overwrite=True
    )

    tuner.search_space_summary()

    # Run Search
    print("\n--- Starting Hyperparameter Search ---")
    tuner.search(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
    )

    # Get Best Hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n--- Best Hyperparameters ---")
    print(f"Conv Filters: {best_hps.get('conv_filters')}")
    print(f"Conv Kernel: {best_hps.get('conv_kernel')}")
    print(f"LSTM Units: {best_hps.get('lstm_units')}")
    print(f"Dense Units: {best_hps.get('dense_units')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")

    # Retrain Best Model
    print("\n--- Retraining Best Model ---")
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    )

    # Evaluate on Test Set
    print("\n--- Evaluating Best Model on Test Set ---")
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test MSE: {loss:.2f}")
    print(f"Test MAE: {mae:.2f}")

    # Save Best Model
    model_save_path = 'rul_model_tuned.weights.h5'
    model.save_weights(model_save_path)
    print(f"Best model weights saved to {model_save_path}")

if __name__ == '__main__':
    main()
