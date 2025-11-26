import numpy as np
from src.data_preprocessing import load_and_preprocess
from src.ge_main import run_ge
from src.evaluation import evaluate_expression
from src.visualisation import plot_results

X_train, X_test, y_train, y_test = load_and_preprocess('data/houses.csv')

best_genome, best_expr = run_ge(X_train, y_train, pop_size=500, generations=20, genome_len=100, max_depth=10)

# Use best expression for prediction
y_pred = [evaluate_expression(best_expr, sample) for sample in X_test]
print("Predictions:", y_pred[:5])


plot_results(y_test, y_pred)
