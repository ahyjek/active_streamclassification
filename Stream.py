import numpy as np
import strlearn as sl
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from ActiveLearning import ActiveClassifierVariableUncertaintyStrategy, ActiveClassifierRandomStrategy, ActiveClassifierVariableUncertaintyStrategyWithRandomization
import datetime
import os

streams = [
    sl.streams.StreamGenerator(
        n_chunks=200,
        chunk_size=500,
        n_classes=2,
        n_drifts=1,
        n_features=10,
        random_state=12345
    ),
    sl.streams.StreamGenerator(
        n_chunks=200,
        chunk_size=500,
        n_classes=2,
        n_drifts=2,
        recurring=True,
        n_features=10,
        random_state=12345
    ),
    sl.streams.StreamGenerator(
        n_chunks=200,
        chunk_size=500,
        n_classes=2,
        n_drifts=1,
        concept_sigmoid_spacing=5,
        n_features=10,
        random_state=12345
    ),
    sl.streams.StreamGenerator(
        n_chunks=200,
        chunk_size=500,
        n_classes=2,
        n_drifts=2,
        concept_sigmoid_spacing=5,
        recurring=True,
        n_features=10,
        random_state=12345
    ),
    sl.streams.StreamGenerator(
        n_chunks=200,
        chunk_size=500,
        n_classes=2,
        n_drifts=1,
        concept_sigmoid_spacing=5,
        incremental=True,
        recurring=True,
        n_features=10,
        random_state=12345
    ),
    sl.streams.StreamGenerator(
        n_chunks=200,
        chunk_size=500,
        n_classes=2,
        n_drifts=2,
        concept_sigmoid_spacing=5,
        incremental=True,
        recurring=True,
        n_features=10,
        random_state=12345
    ),
]
# Nazwy rodzajow dryfu
streams_names = [
    "Dryf nagly",
    "Dryf nagly rekurencyjny",
    "Dryf gradualny",
    "Dryf gradualny rekurencyjny",
    "Dryf inkrementalny",
    "Dryf inkrementalny rekurencyjny"
]

# Inicjacja klasyfikatorow
clfs = [
    MLPClassifier(hidden_layer_sizes=(10)),
    ActiveClassifierVariableUncertaintyStrategy(MLPClassifier(hidden_layer_sizes=(10))),
	ActiveClassifierRandomStrategy(MLPClassifier(hidden_layer_sizes=(10))),
	ActiveClassifierVariableUncertaintyStrategyWithRandomization(MLPClassifier(hidden_layer_sizes=(10))),
]
# Nazwy klasyfikatorow
clf_names = [
    "MLP",
    "MLP_ACTIVE_VARIABLE_UNCERTAINTY_STRATEGY",
	"MLP_ACTIVE_VARIABLE_RANDOM_STRATEGY",
	"MLP_ACTIVE_VARIABLE_UNCERTAINTY_STRATEGY_WITH_RANDOMIZATION",
]

# Wybrana metryka
metrics = [
    sl.metrics.f1_score,
    sl.metrics.geometric_mean_score_1
]
# Nazwy metryk
metrics_names = [
    "F1 score",
    "G-mean"
]
				 
# Inicjalizacja ewaluatorow
evaluators = [
    sl.evaluators.TestThenTrain(metrics),
    sl.evaluators.Prequential(metrics)
]
# Nazwy ewaluatorow
evaluators_names = [
    "Test-then-train",
    "Prequential"
]

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


"""
# Uruchomienie
for s, stream in enumerate(streams):
    for e, evaluator in enumerate(evaluators):
        evaluator.process(stream, clfs)
        fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
        for m, metric in enumerate(metrics):
            ax[m].set_title(metrics_names[m] + ", " + streams_names[s] + ", " + evaluators_names[e])
            ax[m].set_ylim(0, 1)
            for i, clf in enumerate(clfs):
                ax[m].plot(evaluator.scores[i, :, m], label=clf_names[i])
            for chart in ax.flat:
                chart.set(xlabel='Chunk', ylabel='Metric')
            ax[m].legend()
        timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = "results/" + streams_names[s] + "_" + evaluators_names[e] + timestr + ".png"
        ensure_dir(filename)
        plt.savefig(filename)
        print (filename + " saved.")
"""

# Uruchomienie Test-then-train
for s, stream in enumerate(streams):
    e = 0
    evaluator = evaluators[e]
    evaluator.process(stream, clfs)
    fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
    for m, metric in enumerate(metrics):
        ax[m].set_title(metrics_names[m] + ", " + streams_names[s] + ", " + evaluators_names[e])
        ax[m].set_ylim(0, 1)
        for i, clf in enumerate(clfs):
            ax[m].plot(evaluator.scores[i, :, m], label=clf_names[i])
        for chart in ax.flat:
            chart.set(xlabel='Chunk', ylabel='Metric')
        ax[m].legend()
    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = "results/" + streams_names[s] + "_" + evaluators_names[e] + timestr + ".png"
    ensure_dir(filename)
    plt.savefig(filename)
    print (filename + " saved.")

"""
# Uruchomienie Prequential
for s, stream in enumerate(streams):
    e = 1
    evaluator = evaluators[e]
    evaluator.process(stream, clfs)
    fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
    for m, metric in enumerate(metrics):
        ax[m].set_title(metrics_names[m] + ", " + streams_names[s] + ", " + evaluators_names[e])
        ax[m].set_ylim(0, 1)
        for i, clf in enumerate(clfs):
            ax[m].plot(evaluator.scores[i, :, m], label=clf_names[i])
        for chart in ax.flat:
            chart.set(xlabel='Chunk', ylabel='Metric')
        ax[m].legend()
    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = "results/" + streams_names[s] + "_" + evaluators_names[e] + timestr + ".png"
    ensure_dir(filename)
    plt.savefig(filename)
    print (filename + " saved.")
"""