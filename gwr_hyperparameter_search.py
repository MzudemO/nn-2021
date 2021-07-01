import torch
import torchvision.transforms as transforms

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.logging import TensorboardLogger, TextLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    timing_metrics,
    cpu_usage_metrics,
    disk_usage_metrics,
    forgetting_metrics,
    confusion_matrix_metrics,
)

import numpy as np

from gwr_strategy import GWRStrategy

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((16, 16)),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ]
)

# class-incremental single task: 1 experience per class
scenario = SplitMNIST(
    n_experiences=10, train_transform=transform, eval_transform=transform
)

tb_logger = TensorboardLogger()
text_logger = TextLogger(open("log.txt", "a"))
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    # resource metrics
    timing_metrics(epoch=True, epoch_running=True),
    cpu_usage_metrics(experience=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    # performance metrics
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    # additional visualization
    confusion_matrix_metrics(save_image=True),
    # loggers
    loggers=[tb_logger, text_logger],
)

# SplitMNIST config - needs some testing to figure out good values
config = {
    "a_threshold": [0.5, 0.5],
    "beta": 0.7,
    "learning_rates": [0.2, 0.001],
    "context": True,
    "num_context": 2,
    "train_replay": True,
    "e_labels": [10, 10],
    "s_labels": [10],
}

thresholds = []
forgetting = []
accuracies = []
grid = zip(np.arange(0.1, 1.0, 0.2), np.arange(0.1, 1.0, 0.2))
for e_t, s_t in grid:
    config["a_threshold"] = [e_t, s_t]
    cl_strategy = GWRStrategy(
        train_mb_size=100,
        train_epochs=1,
        eval_mb_size=50,
        evaluator=eval_plugin,
        config=config,
    )

    print("Starting experiment...")
    results = []
    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary which contains all the metric values
        res = cl_strategy.train(experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        # test also returns a dictionary which contains all the metric values
        results.append(cl_strategy.eval(scenario.test_stream))

    acc = results[-1]["Top1_Acc_Epoch/train_phase/test_stream/Task000"]
    forgetting = results[-1]["StreamForgetting/eval_phase/test_stream"]
    print(
        f"Final evaluation stream for thresholds {e_t}, {s_t}: {acc * 100}% accuracy, {forgetting} forgetting"
    )
    thresholds.append([e_t, s_t])
    accuracies.append(acc)
    forgetting.append(forgetting)

accuracies = np.array(accuracies)
forgetting = np.array(forgetting)

max_acc = np.max(accuracies)
max_acc_parameters = grid[np.argmax(accuracies)]
min_forgetting = np.min(forgetting)
min_forgetting_parameters = grid[np.argmin(forgetting)]

print(grid)
print(accuracies)
print(forgetting)

print("__________")

print(f"Best values for accuracy: {max_acc_parameters} with {max_acc}")
print(f"Best values for forgetting: {min_forgetting_parameters} with {min_forgetting}")
