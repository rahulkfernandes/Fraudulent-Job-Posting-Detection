import re
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

import time

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import NearestNeighbors


FILE_PATH = "../data/job_train.csv"

def load_data(file_path):
    return pd.read_csv(file_path)

def eda(data):
    print('Columns:')
    print(data.columns)

    print('Shape = ', data.shape)

    print(data.describe())

    print(data.head())

    print("Percentage of null values per column:")
    print(data.isnull().sum() / len(data) * 100)
    
    # Categorical Data Analysis
    print('Corrrelaion Matrix:')
    corr_matrix = data[
        ['telecommuting', 'has_questions', 'has_company_logo', 'fraudulent']
        ].corr()
    print(corr_matrix)
    
    # plot_corr(corr_matrix)

    print('\n', data['fraudulent'].value_counts())
    print(data['telecommuting'].value_counts())
    print(data['has_questions'].value_counts())
    print(data['has_company_logo'].value_counts())

    # Text Data Analysis
    # plot_common_word_count(data)
    
# Plot Correlation Matrix
def plot_corr(corr_matrix):
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.show()

def plot_common_word_count(data):
    fraudulent_text = " ".join(
        data[data['fraudulent'] == 1]['description'].dropna().apply(clean_text_eda)
    )
    legit_text = " ".join(
        data[data['fraudulent'] == 0]['description'].dropna().apply(clean_text_eda)
    )

    # Count word frequencies for fraudulent postings
    fraud_word_counts = Counter(fraudulent_text.split())

    # Count word frequencies for legitimate postings
    legit_word_counts = Counter(legit_text.split())

    # Remove stop words from the word counts
    filtered_fraud_word_counts = {
        word: count for word,
        count in fraud_word_counts.items() if word not in ENGLISH_STOP_WORDS
    }
    filtered_legit_word_counts = {
        word: count for word,
        count in legit_word_counts.items() if word not in ENGLISH_STOP_WORDS
    }

    # Get the top 20 most common words
    top_filtered_fraud_words = Counter(filtered_fraud_word_counts).most_common(20)
    top_filtered_legit_words = Counter(filtered_legit_word_counts).most_common(20)
    
    # Fraudulent postings
    fraud_words, fraud_freqs = zip(*top_filtered_fraud_words)

    plt.figure(figsize=(12, 6))
    plt.barh(fraud_words, fraud_freqs, color='red', alpha=0.7)
    plt.title("Top Words in Fraudulent Job Postings (Filtered)", fontsize=16)
    plt.xlabel("Frequency", fontsize=12)
    plt.gca().invert_yaxis()
    plt.show()

    # Legitimate postings
    legit_words, legit_freqs = zip(*top_filtered_legit_words)

    plt.figure(figsize=(12, 6))
    plt.barh(legit_words, legit_freqs, color='green', alpha=0.7)
    plt.title("Top Words in Legitimate Job Postings (Filtered)", fontsize=16)
    plt.xlabel("Frequency", fontsize=12)
    plt.gca().invert_yaxis()
    plt.show()

# Helper function to clean text
def clean_text_eda(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

##### Preprocessing Experimentation #####

# Function to calculate capitalization metrics
def capitalization_metrics(text):
    capitalized_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
    sentences = re.split(r'[.!?]', text)
    capitalized_sentences = len([s.strip() for s in sentences if s.isupper()])
    return capitalized_words, capitalized_sentences

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    # Remove stop words
    filtered_text = ' '.join(
        [word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS]
    )
    return filtered_text

# Function to split text into sentences
def split_into_sentences(text):
    # Split based on punctuation that ends sentences
    return re.split(r'[.!?]', text)

# Function to count syllables in a word
def count_syllables(word):
    vowels = "aeiouy"
    word = word.lower()
    syllables = 0
    previous_char_was_vowel = False

    for char in word:
        if char in vowels:
            if not previous_char_was_vowel:
                syllables += 1
            previous_char_was_vowel = True
        else:
            previous_char_was_vowel = False

    # Adjust for silent 'e' at the end
    if word.endswith("e"):
        syllables -= 1
    # Ensure at least one syllable
    return max(syllables, 1)

# Function to calculate Flesch Reading Ease score
def flesch_reading_ease(text):
    sentences = split_into_sentences(text)
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty strings
    num_sentences = len(sentences)

    # Use CountVectorizer to tokenize words
    vectorizer = CountVectorizer()
    words = vectorizer.build_tokenizer()(text)
    num_words = len(words)

    # Count syllables in all words
    num_syllables = sum(count_syllables(word) for word in words)

    # Avoid division by zero
    if num_sentences == 0 or num_words == 0:
        return 0

    # Calculate averages
    asl = num_words / num_sentences  # Average Sentence Length
    asw = num_syllables / num_words  # Average Syllables per Word

    # Apply Flesch Reading Ease formula
    return 206.835 - (1.015 * asl) - (84.6 * asw)

def preprocessing(data):
    # Drop useless columns and null rows
    data.drop(
        ['title','location', 'requirements'],
        axis=1,
        inplace=True
    )
    data.dropna(subset=['description'], inplace=True)
    
    if (data.isnull().any().sum() > 0):
        print('There are null values!!')
        print("Percentage of null values per column:")
        print(data.isnull().sum() / len(data) * 100)

    # Get capitalization metrics
    data[['capitalized_words', 'capitalized_sentences']] = data['description'].apply(
        lambda x: pd.Series(capitalization_metrics(x))
    )

    # Get Character length
    data['character_length'] = data['description'].apply(len)

    # Add a number of digits column
    data['num_digits'] = data['description'].apply(
        lambda x: sum(c.isdigit() for c in x) if pd.notnull(x) else 0
    )

    data['flesch_reading_ease'] = data['description'].apply(flesch_reading_ease)

    # Clean text data
    data['description'] = data['description'].apply(preprocess_text)

    # Add a word count column
    data['word_count'] = data['description'].apply(lambda x: len(x.split()))

    # Add a unique word ratio column
    data['unique_word_ratio'] = data['description'].apply(
        lambda x: len(set(x.lower().split())) / 
            len(x.split()) if len(x.split()) > 0 else 0
    )

    print('Corrrelaion Matrix After Feature Engineering:')
    corr_matrix = data[
        ['fraudulent', 'has_company_logo', 'capitalized_words',
       'capitalized_sentences', 'character_length', 'num_digits', 'word_count',
       'unique_word_ratio', 'flesch_reading_ease', 'telecommuting', 'has_questions']
        ].corr()
    print(corr_matrix)
    # plot_corr(corr_matrix)

    # Removed due to low correlation
    data.drop(
        ['capitalized_sentences','num_digits', 'flesch_reading_ease', 'word_count'],
        axis=1,
        inplace=True
    )

    # 1. TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(data["description"])

    # 2. Get engineered features
    engineered_features = data[
        ['capitalized_words', 'character_length', 'unique_word_ratio']
    ]

    # 3. Scale the engineered features
    scaler = StandardScaler()
    engineered_features_scaled = scaler.fit_transform(engineered_features)

    # 4. Get binary features
    binary_features = data[["has_questions"]]

    # 5. Convert binary features to sparse matrix
    binary_features_sparse = np.array(binary_features).astype(float)

    # 6. Combine all features (TF-IDF + standardized engineered features + binary features)
    # Convert the binary features to sparse format to maintain sparse matrix compatibility
    binary_features_sparse = np.array(binary_features_sparse)

    # Combine TF-IDF with standardized engineered features and binary features
    X_combined = hstack([
        X_tfidf,
        engineered_features_scaled,
        binary_features_sparse
    ])

    y_labels = data["fraudulent"]

    print('Shape after preprocesing =', X_combined.shape)

    feature_names = [
        'description',
        'capitalized_words',
        'character_length',
        'unique_word_ratio', 
        'has_questions'
    ]
    return X_combined, y_labels, feature_names

def smote(X, y, target_class, n_samples, k_neighbors=5, random_state=42):
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
        alpha = np.random.rand()  # Random value between 0 and 1
        synthetic_sample = sample + alpha * (neighbor - sample)
        synthetic_samples.append(synthetic_sample)
    
    # Combine the original data with the synthetic samples
    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.array([target_class] * n_samples)
    
    X_resampled = np.vstack((X, X_synthetic))
    y_resampled = np.hstack((y, y_synthetic))
    
    return X_resampled, y_resampled

def balance_data(X, y, strategy='smote'):

    # Combine features and labels
    data = pd.DataFrame(X)
    data['fraudulent'] = y.values if hasattr(y, 'values') else y

    # Separate the majority and minority classes
    majority_class = data[data['fraudulent'] == 0]
    minority_class = data[data['fraudulent'] == 1]
    
    if strategy == 'smote':
        # Use SMOTE for synthetic oversampling
        # smote_instance = SMOTE(random_state=42)
        n_samples = len(majority_class)-len(minority_class)
        X_combined, y_combined = smote(
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
            "Method should be either 'smote' or 'rnd_sample'."
        )

    return X_combined, y_combined

def get_importance(model, X, col_names):
    # Get feature importances
    feature_importances = model.feature_importances_

    # Combine feature names with their importance values
    # Get feature names for TF-IDF features
    tfidf_feature_names = ['TF-IDF_' + str(i) for i in range(X.shape[1] - len(col_names) - 3)]

    # Get the combined feature names (TF-IDF + engineered + binary)
    combined_feature_names = tfidf_feature_names + col_names[1:]  # Adjust the col_names list to match
    
    # Sort the feature importances and print the top ones
    feature_importance_data = list(zip(combined_feature_names, feature_importances))
    feature_importance_data = sorted(feature_importance_data, key=lambda x: x[1], reverse=True)

    print("\nFeature Importance (last 10):")
    for feature, importance in feature_importance_data[-10:]:  # Show last 10 features
        print(f"{feature}: {importance:.4f}")

def grid_searcher(X_train, y_train):
    # Define the parameter grid
    
    param_grid_dict = {
        # 'LogisticRegression': {
        #     'C': [0.1, 1.0, 10.0, 100], 
        #     'solver': ['lbfgs', 'liblinear'],
        #     'max_iter': [200, 500, 1000]
        # },
        'RandomForestClassifier': {
            'n_estimators': [200],
            'max_depth': [40]
        },
        'SGDClassifier': {
            "loss": ["log_loss"],
            "penalty": ['l2', 'elasticnet'],
            "alpha": [0.0001, 0.01]
        },
        'MLPClassifier': {
            'hidden_layer_sizes': [(50,2)],  # Layer configurations
            'alpha': [0.0001],  # L2 regularization
            'activation': ['relu'],
            'max_iter': [500]
        }
    }

    # Initialize the models
    models = {
        'MLPClassifier': MLPClassifier(random_state=42),
        # 'SGDClassifier': SGDClassifier(random_state=42),
        # 'LogisticRegression': LogisticRegression(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(random_state=42)
    }

    # Define a dictionary to store the best models and scores
    best_models = {}
    best_scores = {}

    # Perform GridSearchCV for each model
    for model_name, model in models.items():
        print(f'Tuning {model_name}')
        model_start = time.time()
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid_dict[model_name],
            scoring='f1_macro',
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

    # Select the best model based on the highest F1 Macro score
    best_model_name = max(best_scores, key=best_scores.get)
    best_model = best_models[best_model_name]

    print("\nBest Models:", best_models.keys)
    return best_models

def train_eval(X, y, col_names):

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size=0.2,
        random_state=30
    )

    if hasattr(X_train, "toarray"):
        X_train_dense = X_train.toarray()
    else:
        X_train_dense = X_train

    if hasattr(X_test, "toarray"):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test

    X_train_resample, y_train_resample = balance_data(
        X_train_dense, y_train, strategy='smote'
    )
    print('Shape after oversampling =', X_train_resample.shape)


    # best_models = grid_searcher(X_train_resample, y_train_resample)

    # # Train on the best model
    # best_model.fit(X_train_resample, y_train_resample)

    # # Predict on the test set
    # y_pred = best_model.predict(X_test_dense)

    from sklearn.naive_bayes import GaussianNB
    

    ensemble_models = {
        'MLPClassifier': MLPClassifier(
            hidden_layer_sizes=(50, 2), 
            alpha=0.0001, max_iter=500, 
            activation='relu',
            random_state=42
        ), 
        # 'SGDClassifier': SGDClassifier(
        #     loss='log_loss',
        #     random_state=42
        # ), 
        'RandomForestClassifier': RandomForestClassifier(
            max_depth=40,
            n_estimators=200,
            random_state=42
        )
    }

    ensemble = []
    for model in ensemble_models:
        ensemble.append((model, ensemble_models[model]))
        
    # Create a voting classifier
    voting_clf = VotingClassifier(estimators=ensemble, voting='soft', verbose=4)

    # Train the model
    voting_clf.fit(X_train_resample, y_train_resample)

    y_pred = voting_clf.predict(X_test_dense)

    # Print classification report
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    start = time.time()
    data_set = load_data(FILE_PATH)
    print('\n =========================EDA==========================')
    eda(data_set)

    print('\n =====================Preprocessing====================')
    features, labels, cols = preprocessing(data_set)

    print('\n =======================Training=======================')
    train_eval(features, labels, cols)

    runtime = (time.time() - start) / 60.0
    print('Total runtime =',runtime)