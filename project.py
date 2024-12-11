import re
import time
import numpy as np
import pandas as pd
from sklearn.utils import resample
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score, make_scorer


class my_model():
    def __init__(self):
        self.model = None
        self.ensemble_models = {
            'MLPClassifier': MLPClassifier(
                hidden_layer_sizes = (50,50),  # (50,50) (128,128)
                alpha = 0.0001,
                max_iter = 500, 
                activation ='relu',
                random_state = 42,
                verbose = True
            ), 
            'RandomForestClassifier': RandomForestClassifier(
                max_depth = 40,
                n_estimators = 100,
                random_state = 42,
                verbose = True
            ),
            'LogisticRegression': LogisticRegression(
                C = 100, 
                solver = 'liblinear',
                max_iter = 100,
                random_state = 42,
                verbose = True
            ),
            # 'SGDClassifier': CalibratedClassifierCV(SGDClassifier(
            #     loss = 'hinge',
            #     penalty = 'l2',
            #     alpha = 0.0001,
            #     max_iter = 1000
            # ), method='sigmoid')
        }
        self.vectorizer = TfidfVectorizer()
        self.scaler = StandardScaler()


    def _preprocessing(self, X, run_type):
        
        X_set = X.drop(['telecommuting', 'requirements', 'location'], axis=1)
        
        if (X_set.isnull().any().sum() > 0):
            print('There are null values!!')
            print("Percentage of null values per column:")
            print(X_set.isnull().sum() / len(X_set) * 100)

        def _capitalization_metrics(text):
            capitalized_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
            return capitalized_words
        
        # Get description character length
        X_set['desc_character_length'] = X_set['description'].apply(len)

        # Combine text columns into a single feature
        X_set['description'] = X_set['title'] + ' ' + X_set['description']

        # Get word capitalization metric 
        X_set['capitalized_words'] = X_set['description'].apply(
            lambda x: pd.Series(_capitalization_metrics(x))
        )
        
        def _clean_text(text):
            text = text.lower()  # Convert to lowercase
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            text = re.sub(r'\d+', '', text)  # Remove numbers
            # Remove stop words
            filtered_text = ' '.join(
                [word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS]
            )
            return filtered_text

        # Clean text data
        X_set['description'] = X_set['description'].apply(_clean_text)

        # Add a unique word ratio column
        X_set['unique_word_ratio'] = X_set['description'].apply(
            lambda x: len(set(x.lower().split())) / 
                len(x.split()) if len(x.split()) > 0 else 0
        )

        if run_type == 'train':            
            # 1. Get engineered features for scaling
            engineered_features = X_set[[
                'capitalized_words',
                'desc_character_length', 
                'unique_word_ratio'
            ]]
            # 2. TF-IDF Vectorization
            X_tfidf = self.vectorizer.fit_transform(X_set['description'])
            # 3. Scale the engineered features  
            engineered_features_scaled = self.scaler.fit_transform(engineered_features)
                   
        elif run_type == 'test':
            # 1. Get engineered features for scaling
            engineered_features = X_set[[
                'capitalized_words',
                'desc_character_length',
                'unique_word_ratio'
            ]]
            # 2. TF-IDF Vectorization
            X_tfidf = self.vectorizer.transform(X_set['description'])
            # 3. Scale the engineered features
            engineered_features_scaled = self.scaler.transform(engineered_features)
            # country_encoded = self.one_hot_encoder.transform(X_set[['country']])
        
        else:
            raise ValueError(
                "run_type should be either 'train' or 'test'."
            )

        # 4. Get binary features
        binary_features = X_set[['has_questions', 'has_company_logo']]

        # 5. Combine all features and convert to sparse
        # (TF-IDF + standardized engineered features + binary features)
        X_combined = hstack([
            X_tfidf,
            engineered_features_scaled,
            csr_matrix(binary_features.astype(float))
        ])

        # Convert to dense matrix
        if hasattr(X_combined, "toarray"):
            X_dense =  X_combined.toarray()
        else:
            X_dense =  X_combined

        return X_dense
    
    def _smote(self, X, y, target_class, n_samples, k_neighbors=5, random_state=42):
        np.random.seed(random_state)
        
        # Extract minority class samples
        X_minority = X[y == target_class]
        
        # Fit k-nearest neighbors
        knn = NearestNeighbors(n_neighbors=k_neighbors)
        knn.fit(X_minority)
        
        # Find k-nearest neighbors for each minority sample
        neighbors = knn.kneighbors(X_minority, return_distance=False)
        
        # Generate synthetic samples
        synthetic_samples = []
        for _ in range(n_samples):
            # Randomly select a minority sample
            idx = np.random.randint(0, X_minority.shape[0])
            sample = X_minority[idx]
            
            # Randomly select one of its k-nearest neighbors
            neighbor_idx = np.random.choice(neighbors[idx][1:])
            neighbor = X_minority[neighbor_idx]
            
            # Generate a synthetic sample along the line between sample and neighbor
            #alpha = np.random.rand()  # Random value between 0 and 1
            rng = np.random.default_rng(random_state)
            idx = rng.integers(0, X_minority.shape[0])
            alpha = rng.random()

            synthetic_sample = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic_sample)
        
        # Combine the original data with the synthetic samples
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.array([target_class] * n_samples)
        
        X_resampled = np.vstack((X, X_synthetic))
        y_resampled = np.hstack((y, y_synthetic))
        
        return X_resampled, y_resampled

    def _balance_data(self, X, y, strategy='smote'):

        # Combine features and labels
        data = pd.DataFrame(X)
        data['fraudulent'] = y.values if hasattr(y, 'values') else y

        # Separate the majority and minority classes
        majority_class = data[data['fraudulent'] == 0]
        minority_class = data[data['fraudulent'] == 1]
        
        if strategy == 'smote':
            # Use SMOTE for synthetic oversampling
            n_samples = len(majority_class)-len(minority_class)
            X_combined, y_combined = self._smote(
                X,
                y.values,
                1,
                n_samples=n_samples,
                k_neighbors=7
            )

        elif strategy == 'rnd_sample':
            # Oversample the minority class
            minority_class_oversampled = resample(
                minority_class,
                replace=True,    # Sample with replacement
                n_samples=len(majority_class),  # Match majority class size
                random_state=42   # Ensure reproducibility
            )

            # Combine the majority class with the oversampled minority class
            oversampled_data = pd.concat([majority_class, minority_class_oversampled])

            # Separate the features and labels again
            X_combined = oversampled_data.drop('fraudulent', axis=1)
            y_combined = oversampled_data['fraudulent']
        
        else:
            raise ValueError(
                "strategy should be either 'smote' or 'rnd_sample'."
            )

        return X_combined, y_combined

    def _grid_searcher(self, X_train, y_train):
        # Define the parameter grid
        
        param_grid_dict = {
            'LogisticRegression': {
                'C': [0.1, 1, 10, 100], 
                'solver': ['lbfgs', 'liblinear'],
                'max_iter': [100, 200]
            },
            'RandomForestClassifier': {
                'n_estimators': [100, 200],
                'max_depth': [30, 40]
            },
            'SGDClassifier': {
                "loss": ["log_loss", 'hinge'],
                "penalty": ['l2'],
                "alpha": [0.0001, 0.001],
                'max_iter': [1000, 2000]
            },
            'MLPClassifier': {
                'hidden_layer_sizes': [(32,32), (50,50)],  # Layer configurations
                'alpha': [0.0001],  # L2 regularization
                'activation': ['relu'],
                'max_iter': [500]
            }
        }

        # Initialize the models
        models = {
            'MLPClassifier': MLPClassifier(random_state=42),
            'SGDClassifier': SGDClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42),
            'RandomForestClassifier': RandomForestClassifier(random_state=42)
        }

        # Define a dictionary to store the best models and scores
        best_models = {}
        best_scores = {}

        f1_minority_scorer = make_scorer(f1_score, average='binary', pos_label=1)

        # Perform GridSearchCV for each model
        for model_name, model in models.items():
            print(f'Tuning {model_name}')
            model_start = time.time()
            
            # Initialize GridSearchCV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid_dict[model_name],
                scoring=f1_minority_scorer,
                cv=3,
                verbose=3
            )

            # Fit GridSearchCV
            grid_search.fit(X_train, y_train)

            # Store the best model and score
            best_models[model_name] = grid_search.best_estimator_
            best_scores[model_name] = grid_search.best_score_

            print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best F1 Macro Score for {model_name}: {grid_search.best_score_}")
            model_runtime = (time.time() - model_start) / 60.0
            print(f'{model_name} runtime =', model_runtime)

        # Sort the models by score in descending order
        sorted_best_models = {
        model_name: best_models[model_name]
        for model_name in sorted(best_scores, key=best_scores.get, reverse=True)
        }
        print('\nBest Models Sorted by F1 Macro Score:', sorted_best_models)
        
        # Select the best model based on the highest F1 Macro score
        best_model_name = max(best_scores, key=best_scores.get)
        best_model = best_models[best_model_name]

        print("\nBest Model:", best_model)
        return best_model

    def fit(self, X, y):
        # do not exceed 29 mins

        X_processed = self._preprocessing(X, run_type='train')
        print(f"Original X shape: {X.shape}")
        print(f"Processed X shape: {X_processed.shape}")
        
        X_resample, y_resample = self._balance_data(
            X_processed,  y, strategy='smote'
        )
        print('Shape after oversampling =', X_resample.shape)

        # self.model = self._grid_searcher(X_resample, y_resample)

        ensemble = []
        for model in self.ensemble_models:
            ensemble.append((model, self.ensemble_models[model]))
    
        # Create a voting classifier
        self.model = VotingClassifier(estimators=ensemble, voting='soft', verbose=3)

        # Train the model
        self.model.fit(X_resample, y_resample)

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        
        X_processed = self._preprocessing(X, run_type='test')
        print(f"Original X shape: {X.shape}")
        print(f"Processed X shape: {X_processed.shape}")

        predictions = self.model.predict(X_processed)
        return predictions