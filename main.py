import numpy as np, time, logging, multiprocessing
from src.data_preprocessing import load_and_preprocess
from src.ge_main import run_ge
from src.visualisation import plot_results
from src.models import EvolutionConfig
from src.evaluation import eval_tree, evaluate_top_individuals_on_test

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    logger.info("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess('data/houses.csv')

    start = time.perf_counter()
    cfg = EvolutionConfig("config.json")

    # we want this to return the best 10 genomes and their trees so we can validate on test set
    best_ten = run_ge(X_train, y_train, cfg)

    # Evaluate all top 10 individuals on test dataset
    test_results = evaluate_top_individuals_on_test(best_ten, X_test, y_test, cfg)

    # Use best expression for prediction (for visualization)
    y_pred = [
        eval_tree(
            best_ten[0]['phenotype'],
            {k: row[i] for i, k in enumerate(cfg.feature_names)}
        )
        for row in X_test
    ]
    logger.info("\nPredictions (first 5): %s", y_pred[:5])

    plot_results(y_test, y_pred)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
