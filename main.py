import numpy as np, time
from src.data_preprocessing import load_and_preprocess
from src.ge_main import run_ge
from src.visualisation import plot_results
from src.models import EvolutionConfig
from src.evaluation import eval_tree

X_train, X_test, y_train, y_test = load_and_preprocess('data/houses.csv')
start = time.perf_counter()
cfg = EvolutionConfig("config.json")

# we want this to return the best 10 genomes and their trees so we can validate on test set
best_genome, best_tree = run_ge(X_train, y_train, cfg)

# Use best expression for prediction
# Build predictions on test set using the best tree
y_pred = [
	eval_tree(best_tree, {k: row[i] for i, k in enumerate(cfg.feature_names)})
	for row in X_test
]
print("Predictions:", y_pred[:5])

total = time.perf_counter() - start
avg = total / max(1, cfg.generations)
print(f"Completed {cfg.generations} generations in {total:.4f}s (avg {avg:.6f}s/gen)")

plot_results(y_test, y_pred)
