import sys
import numpy as np
import pandas as pd
import nltk
import sqlite3
import sqlalchemy
import io
import warnings
import pickle

from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator


from xgboost import XGBClassifier
from tabulate import tabulate

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from typing import Tuple, Optional, List



def load_data(database_filepath: str) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame], Optional[pd.Index]]:
    """
    Load data from a SQL database.

    Args:
        database_filepath - str: Path to the database file.

    Returns:
        Tuple[Optional[pd.Series], Optional[pd.DataFrame], Optional[pd.Index]]: 
            X (messages), Y (categories), and category names.
            Returns (None, None, None) if an error occurs.
    """
    try:
        engine = create_engine(f'sqlite:///{database_filepath}')
        df = pd.read_sql_table('chrismo', con=engine)
        X = df['message']
        Y = df.iloc[:,4:]
        category_names = Y.columns
        return X, Y, category_names
    except Exception as e:
        print(f"Error loading data: {e}") #error message if any 
        return None, None, None


def tokenize(text: str) -> List[str]:
    """
    Tokeinze and process text data.

    Args:
        text (str): Input text to be tokenized.

    Returns:
        List[str]: List of processed tokens.
    """
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    STOPWORDS = stopwords.words("english")
    word_tokens = [word for word in word_tokens if word not in STOPWORDS]   #stop word removal
    
    tokens = []
    for word_token in word_tokens:
        token = lemmatizer.lemmatize(word_token.lower().strip(), pos='v')        #pos tagging
        tokens.append(token)
        
    return tokens



def build_model() -> Pipeline:
    """
    Builds and return to train tokenized texts on for our classification
    
    Args:
        None
    
    Returns:
        Ppieline: SK Learn object containing model
    
    """
    pipeline = Pipeline([
                    ('features', FeatureUnion([
                        ('text_pipeline', Pipeline([
                            ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                            ('tfidf_transformer', TfidfTransformer())
                        ])),
                    ])),
                    ('classifier', MultiOutputClassifier(XGBClassifier(
                        max_depth=3,
                        learning_rate=0.1,
                        n_estimators=200,
                        use_label_encoder=False, 
                        eval_metric='mlogloss',
                    )))
                ])
    return pipeline



def evaluate_model(model: BaseEstimator, X_test: pd.Series, Y_test: pd.DataFrame, category_names: pd.Index) -> pd.DataFrame:
    """
    Evaluate the model and print detailed metrics for each category.

    Args:
        model (BaseEstimator): Trained model to evaluate.
        X_test (pd.Series): Test features.
        Y_test (pd.DataFrame): True labels for test data.
        category_names (pd.Index): Names of the categories.

    Returns:
        pd.DataFrame: Overall evaluation metrics for each category.
    """
    
    y_pred_grid = model.predict(X_test)
    y_test_df = pd.DataFrame(Y_test, columns=category_names)
    predictions_df = pd.DataFrame(y_pred_grid, columns=category_names)

    print("## Detailed Metrics Breakdown by Category")

    overall_metrics = []

    for column in category_names:
        print(f"\n### {column}")
        report = classification_report(y_test_df[column], predictions_df[column], output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        class_labels = sorted(set(df_report.index) - {'accuracy', 'macro avg', 'weighted avg'})
        df_report = df_report.loc[class_labels + ['weighted avg'], ['precision', 'recall', 'f1-score', 'support']].round(3)

        accuracy = report['accuracy']

        df_report.loc['accuracy'] = [accuracy, accuracy, accuracy, df_report.loc['weighted avg', 'support']]

        print(tabulate(df_report, headers='keys', tablefmt='pipe', showindex=True))

        weighted_avg = df_report.loc['weighted avg']
        overall_metrics.append({
            'Category': column,
            'Precision': weighted_avg['precision'],
            'Recall': weighted_avg['recall'],
            'F1-score': weighted_avg['f1-score'],
            'Accuracy': accuracy
        })

    overall_df = pd.DataFrame(overall_metrics).set_index('Category')
    return overall_df



def save_model(model: BaseEstimator, model_filepath: str) -> None:
    """
    Save our trained model to pickke file.

    Args:
        model (BaseEstimator): Trained model to save.
        model_filepath (str): Path where the model will be saved.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


        
        
def main() -> None:
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluation_results = evaluate_model(model, X_test, Y_test, category_names)
        print("Overall evaluation results:")
        print(evaluation_results)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()