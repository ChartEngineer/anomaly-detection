"""
Anomaly Detection Algorithms Module

This module implements various anomaly detection algorithms:
- Isolation Forest
- One-Class SVM
- Autoencoder (Neural Network)
- Local Outlier Factor
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """
    A class implementing multiple anomaly detection algorithms.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def isolation_forest(self, contamination=0.1, n_estimators=100):
        """
        Initialize Isolation Forest model.
        
        Args:
            contamination (float): Expected proportion of anomalies
            n_estimators (int): Number of trees
            
        Returns:
            IsolationForest: Configured model
        """
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        return model
    
    def one_class_svm(self, nu=0.1, kernel='rbf', gamma='scale'):
        """
        Initialize One-Class SVM model.
        
        Args:
            nu (float): Upper bound on fraction of training errors
            kernel (str): Kernel type
            gamma (str or float): Kernel coefficient
            
        Returns:
            OneClassSVM: Configured model
        """
        model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma
        )
        return model
    
    def local_outlier_factor(self, n_neighbors=20, contamination=0.1):
        """
        Initialize Local Outlier Factor model.
        
        Args:
            n_neighbors (int): Number of neighbors
            contamination (float): Expected proportion of anomalies
            
        Returns:
            LocalOutlierFactor: Configured model
        """
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True  # For prediction on new data
        )
        return model
    
    def create_autoencoder(self, input_dim, encoding_dim=10, hidden_layers=[50, 20]):
        """
        Create an autoencoder neural network for anomaly detection.
        
        Args:
            input_dim (int): Input dimension
            encoding_dim (int): Encoding dimension
            hidden_layers (list): Hidden layer sizes
            
        Returns:
            keras.Model: Autoencoder model
        """
        # Input layer
        input_layer = keras.Input(shape=(input_dim,))
        
        # Encoder
        encoded = input_layer
        for units in hidden_layers:
            encoded = layers.Dense(units, activation='relu')(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu', name='encoded')(encoded)
        
        # Decoder
        decoded = encoded
        for units in reversed(hidden_layers):
            decoded = layers.Dense(units, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Create autoencoder
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def train_autoencoder(self, X_train, X_val, input_dim, epochs=50, batch_size=32):
        """
        Train autoencoder model.
        
        Args:
            X_train (np.array): Training data
            X_val (np.array): Validation data
            input_dim (int): Input dimension
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            
        Returns:
            tuple: (trained_model, history)
        """
        # Create autoencoder
        autoencoder = self.create_autoencoder(input_dim)
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        return autoencoder, history
    
    def predict_autoencoder_anomalies(self, autoencoder, X, threshold_percentile=95):
        """
        Predict anomalies using autoencoder reconstruction error.
        
        Args:
            autoencoder: Trained autoencoder model
            X (np.array): Input data
            threshold_percentile (float): Percentile for threshold
            
        Returns:
            tuple: (predictions, reconstruction_errors)
        """
        # Get reconstructions
        reconstructions = autoencoder.predict(X)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
        
        # Set threshold based on percentile
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        
        # Predict anomalies
        predictions = (reconstruction_errors > threshold).astype(int)
        
        return predictions, reconstruction_errors
    
    def train_traditional_models(self, X_train, y_train, X_val, y_val):
        """
        Train traditional anomaly detection models.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
            X_val (np.array): Validation features
            y_val (np.array): Validation labels
            
        Returns:
            dict: Trained models and their performance
        """
        results = {}
        
        # Isolation Forest
        print("Training Isolation Forest...")
        iso_forest = self.isolation_forest(contamination=0.1)
        iso_forest.fit(X_train)
        
        # Predict on validation set
        iso_pred = iso_forest.predict(X_val)
        iso_pred = np.where(iso_pred == -1, 1, 0)  # Convert -1 to 1 (anomaly)
        
        iso_scores = iso_forest.decision_function(X_val)
        iso_auc = roc_auc_score(y_val, -iso_scores)  # Negative because lower scores indicate anomalies
        
        results['isolation_forest'] = {
            'model': iso_forest,
            'predictions': iso_pred,
            'scores': iso_scores,
            'auc': iso_auc
        }
        
        # One-Class SVM
        print("Training One-Class SVM...")
        oc_svm = self.one_class_svm(nu=0.1)
        oc_svm.fit(X_train)
        
        svm_pred = oc_svm.predict(X_val)
        svm_pred = np.where(svm_pred == -1, 1, 0)  # Convert -1 to 1 (anomaly)
        
        svm_scores = oc_svm.decision_function(X_val)
        svm_auc = roc_auc_score(y_val, -svm_scores)  # Negative because lower scores indicate anomalies
        
        results['one_class_svm'] = {
            'model': oc_svm,
            'predictions': svm_pred,
            'scores': svm_scores,
            'auc': svm_auc
        }
        
        # Local Outlier Factor
        print("Training Local Outlier Factor...")
        lof = self.local_outlier_factor(contamination=0.1)
        lof.fit(X_train)
        
        lof_pred = lof.predict(X_val)
        lof_pred = np.where(lof_pred == -1, 1, 0)  # Convert -1 to 1 (anomaly)
        
        lof_scores = lof.decision_function(X_val)
        lof_auc = roc_auc_score(y_val, -lof_scores)  # Negative because lower scores indicate anomalies
        
        results['local_outlier_factor'] = {
            'model': lof,
            'predictions': lof_pred,
            'scores': lof_scores,
            'auc': lof_auc
        }
        
        return results
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Evaluate model performance.
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\n{model_name} Performance:")
        print("="*50)
        print(classification_report(y_true, y_pred))
        print(f"Confusion Matrix:\n{cm}")
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score']
        }
    
    def compare_models(self, results, y_val):
        """
        Compare performance of different models.
        
        Args:
            results (dict): Model results
            y_val (np.array): Validation labels
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for model_name, result in results.items():
            metrics = self.evaluate_model(y_val, result['predictions'], model_name)
            
            comparison_data.append({
                'Model': model_name,
                'AUC': result['auc'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AUC', ascending=False)
        
        print("\nModel Comparison:")
        print("="*70)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        return comparison_df
    
    def save_models(self, results, file_prefix='anomaly_model'):
        """
        Save trained models to disk.
        
        Args:
            results (dict): Model results
            file_prefix (str): Prefix for saved files
        """
        for model_name, result in results.items():
            file_path = f'models/{file_prefix}_{model_name}.pkl'
            joblib.dump(result['model'], file_path)
            print(f"Saved {model_name} to {file_path}")
    
    def training_pipeline(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Complete training pipeline for all anomaly detection models.
        
        Args:
            X_train, X_val, X_test: Feature sets
            y_train, y_val, y_test: Label sets
            
        Returns:
            dict: All model results
        """
        print("Starting anomaly detection training pipeline...")
        
        # Train traditional models
        traditional_results = self.train_traditional_models(X_train, y_train, X_val, y_val)
        
        # Train autoencoder
        print("Training Autoencoder...")
        autoencoder, history = self.train_autoencoder(X_train, X_val, X_train.shape[1])
        
        # Predict with autoencoder
        ae_pred, ae_errors = self.predict_autoencoder_anomalies(autoencoder, X_val)
        ae_auc = roc_auc_score(y_val, ae_errors)
        
        traditional_results['autoencoder'] = {
            'model': autoencoder,
            'predictions': ae_pred,
            'scores': ae_errors,
            'auc': ae_auc,
            'history': history
        }
        
        # Compare all models
        comparison_df = self.compare_models(traditional_results, y_val)
        
        # Save models
        self.save_models(traditional_results)
        
        # Test best model on test set
        best_model_name = comparison_df.iloc[0]['Model']
        best_result = traditional_results[best_model_name]
        
        print(f"\nTesting best model ({best_model_name}) on test set...")
        
        if best_model_name == 'autoencoder':
            test_pred, test_errors = self.predict_autoencoder_anomalies(best_result['model'], X_test)
        else:
            test_pred_raw = best_result['model'].predict(X_test)
            test_pred = np.where(test_pred_raw == -1, 1, 0)
        
        test_metrics = self.evaluate_model(y_test, test_pred, f"{best_model_name} (Test Set)")
        
        self.results = traditional_results
        self.results['comparison'] = comparison_df
        self.results['test_metrics'] = test_metrics
        
        print("Anomaly detection training pipeline completed!")
        return self.results

def main():
    """
    Main function to run anomaly detection training.
    """
    # Load processed data
    try:
        import joblib
        data = joblib.load('data/processed/network_processed.pkl')
        X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
        y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
        
        print(f"Loaded processed data:")
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
    except FileNotFoundError:
        print("Processed data not found. Please run data_loader.py first.")
        return
    
    # Initialize detector
    detector = AnomalyDetector()
    
    # Run training pipeline
    results = detector.training_pipeline(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("Anomaly detection training completed successfully!")

if __name__ == "__main__":
    main()

