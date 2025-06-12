"""
Data Loader Module for Anomaly Detection System

This module handles:
- Loading and preprocessing network traffic data
- Generating synthetic anomaly data for demonstration
- Data validation and cleaning
- Feature scaling and normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

class AnomalyDataLoader:
    """
    A class for loading and preprocessing data for anomaly detection.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def generate_network_data(self, n_samples=10000, n_features=20, contamination=0.1):
        """
        Generate synthetic network traffic data with anomalies.
        
        Args:
            n_samples (int): Number of samples to generate
            n_features (int): Number of features
            contamination (float): Proportion of anomalies
            
        Returns:
            tuple: (X, y) where X is features and y is labels (0=normal, 1=anomaly)
        """
        np.random.seed(self.random_state)
        
        # Generate normal network traffic patterns
        normal_samples = int(n_samples * (1 - contamination))
        anomaly_samples = n_samples - normal_samples
        
        # Normal traffic features
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features),
            size=normal_samples
        )
        
        # Anomalous traffic features (shifted distribution)
        anomaly_data = np.random.multivariate_normal(
            mean=np.ones(n_features) * 3,  # Shifted mean
            cov=np.eye(n_features) * 2,    # Different covariance
            size=anomaly_samples
        )
        
        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([np.zeros(normal_samples), np.ones(anomaly_samples)])
        
        # Shuffle the data
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        # Create feature names
        feature_names = [
            'packet_size', 'duration', 'src_bytes', 'dst_bytes', 'protocol_type',
            'service', 'flag', 'src_port', 'dst_port', 'tcp_flags',
            'urgent_count', 'hot_count', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root',
            'num_file_creations', 'num_shells'
        ]
        
        # Ensure we have the right number of feature names
        if len(feature_names) < n_features:
            feature_names.extend([f'feature_{i}' for i in range(len(feature_names), n_features)])
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names[:n_features])
        df['label'] = y
        
        print(f"Generated {n_samples} network traffic samples")
        print(f"Normal samples: {normal_samples}, Anomaly samples: {anomaly_samples}")
        print(f"Contamination rate: {contamination:.2%}")
        
        return df
    
    def load_kdd_cup_data(self, file_path=None):
        """
        Load KDD Cup 99 dataset (if available) or generate similar synthetic data.
        
        Args:
            file_path (str): Path to KDD Cup dataset
            
        Returns:
            pd.DataFrame: Loaded or generated data
        """
        if file_path and os.path.exists(file_path):
            try:
                # Load real KDD Cup data
                df = pd.read_csv(file_path)
                print(f"Loaded KDD Cup data from {file_path}")
                return df
            except Exception as e:
                print(f"Error loading KDD Cup data: {e}")
        
        # Generate synthetic data similar to KDD Cup
        print("Generating synthetic network intrusion data...")
        return self.generate_network_data(n_samples=50000, n_features=41, contamination=0.2)
    
    def preprocess_data(self, df, target_column='label'):
        """
        Preprocess the data for anomaly detection.
        
        Args:
            df (pd.DataFrame): Input data
            target_column (str): Name of the target column
            
        Returns:
            tuple: (X_scaled, y, feature_names)
        """
        # Separate features and target
        X = df.drop(columns=[target_column, 'timestamp'], errors='ignore')
        y = df[target_column]
        
        # Handle categorical variables (if any)
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = self.label_encoder.fit_transform(X[col].astype(str))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Preprocessed data shape: {X_scaled.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X_scaled, y.values, X.columns.tolist()
    
    def create_time_series_anomalies(self, n_samples=5000, n_features=10):
        """
        Create time series data with temporal anomalies.
        
        Args:
            n_samples (int): Number of time steps
            n_features (int): Number of features
            
        Returns:
            pd.DataFrame: Time series data with anomalies
        """
        np.random.seed(self.random_state)
        
        # Generate time index
        time_index = pd.date_range('2023-01-01', periods=n_samples, freq='H')
        
        # Generate normal time series patterns
        t = np.arange(n_samples)
        data = {}
        
        for i in range(n_features):
            # Create different patterns for each feature
            if i % 3 == 0:
                # Sinusoidal pattern
                signal = 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 1, n_samples)
            elif i % 3 == 1:
                # Linear trend with noise
                signal = 0.01 * t + np.random.normal(0, 2, n_samples)
            else:
                # Random walk
                signal = np.cumsum(np.random.normal(0, 0.5, n_samples))
            
            data[f'feature_{i}'] = signal
        
        df = pd.DataFrame(data, index=time_index)
        
        # Inject anomalies
        anomaly_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
        labels = np.zeros(n_samples)
        labels[anomaly_indices] = 1
        
        # Make anomalies more extreme
        for idx in anomaly_indices:
            for col in df.columns:
                df.loc[df.index[idx], col] *= np.random.uniform(3, 5)
        
        df['label'] = labels
        df['timestamp'] = time_index
        
        print(f"Generated time series data with {len(anomaly_indices)} anomalies")
        return df
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Split data into train, validation, and test sets.
        
        Args:
            X (np.array): Features
            y (np.array): Labels
            test_size (float): Test set proportion
            val_size (float): Validation set proportion
            
        Returns:
            tuple: Train, validation, and test sets
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state, stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_data(self, df, file_path):
        """
        Save processed data to CSV file.
        
        Args:
            df (pd.DataFrame): Data to save
            file_path (str): Output file path
        """
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    
    def load_and_preprocess_pipeline(self, data_type='network', save_processed=True):
        """
        Complete data loading and preprocessing pipeline.
        
        Args:
            data_type (str): Type of data to generate ('network' or 'timeseries')
            save_processed (bool): Whether to save processed data
            
        Returns:
            tuple: Processed data splits
        """
        print("Starting data loading and preprocessing pipeline...")
        
        # Generate or load data
        if data_type == 'network':
            df = self.generate_network_data(n_samples=20000, n_features=20, contamination=0.1)
        elif data_type == 'timeseries':
            df = self.create_time_series_anomalies(n_samples=10000, n_features=15)
        else:
            raise ValueError("data_type must be 'network' or 'timeseries'")
        
        # Save raw data
        if save_processed:
            self.save_data(df, f'data/raw/{data_type}_data.csv')
        
        # Preprocess data
        X, y, feature_names = self.preprocess_data(df)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Save processed data
        if save_processed:
            processed_data = {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
                'feature_names': feature_names
            }
            
            import joblib
            joblib.dump(processed_data, f'data/processed/{data_type}_processed.pkl')
            print(f"Processed data saved to data/processed/{data_type}_processed.pkl")
        
        print("Data loading and preprocessing completed!")
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

def main():
    """
    Main function to run data loading and preprocessing.
    """
    loader = AnomalyDataLoader()
    
    # Load and preprocess network data
    print("Processing network anomaly data...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = loader.load_and_preprocess_pipeline(
        data_type='network'
    )
    
    print(f"Network data processed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Feature names: {feature_names[:5]}...")  # Show first 5 features
    
    # Load and preprocess time series data
    print("\nProcessing time series anomaly data...")
    X_train_ts, X_val_ts, X_test_ts, y_train_ts, y_val_ts, y_test_ts, feature_names_ts = loader.load_and_preprocess_pipeline(
        data_type='timeseries'
    )
    
    print(f"Time series data processed successfully!")
    print(f"Training set shape: {X_train_ts.shape}")

if __name__ == "__main__":
    main()

