import torch
import torchvision.transforms as transforms

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.logging import TensorboardLogger, TextLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, timing_metrics, cpu_usage_metrics, disk_usage_metrics, forgetting_metrics

from gwr_strategy import GWRStrategy

if __name__ == "__main__":
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
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        forgetting_metrics(experience=True, stream=True),
        # loggers
        loggers=[tb_logger, text_logger, interactive_logger],
    )

    cl_strategy = GWRStrategy(
        train_mb_size=500, train_epochs=1, eval_mb_size=100, evaluator=eval_plugin
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
