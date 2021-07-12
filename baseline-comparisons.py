
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import torchvision.transforms as transforms

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.logging import TensorboardLogger, TextLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins import ReplayPlugin, LwFPlugin, EWCPlugin
from avalanche.training.strategies import Naive
from avalanche.models import SimpleMLP
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    timing_metrics,
    cpu_usage_metrics,
    disk_usage_metrics,
    forgetting_metrics,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((16, 16)),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ]
)

scenario = SplitMNIST(
    n_experiences=10, 
    train_transform=transform, 
    eval_transform=transform,
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
    # confusion_matrix_metrics(num_classes=10, stream=True, save_image=True),
    # loggers
    loggers=[text_logger],
)

def get_model():
  return SimpleMLP(num_classes = scenario.n_classes, input_size=256, hidden_layers=1)

for _ in range(5):
  scenario = SplitMNIST(
    n_experiences=10, 
    train_transform=transform, 
    eval_transform=transform,
  )
  
  print("naive")
  naive_model = get_model()
  naive_cl_strategy = Naive(naive_model, SGD(naive_model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), train_mb_size=100, train_epochs=1, eval_mb_size=50, evaluator=eval_plugin)

  results = []
  for i_train, experience in enumerate(scenario.train_stream):
      res = naive_cl_strategy.train(experience)
      results.append(naive_cl_strategy.eval(scenario.test_stream))

  acc = results[-1]["Top1_Acc_Stream/eval_phase/test_stream/Task000"]
  forgetting = results[-1]["StreamForgetting/eval_phase/test_stream"]
  print(
      f"Final evaluation stream: {acc * 100}% accuracy, {forgetting} forgetting"
  )

  print("lwf")
  lwf_model = get_model()
  lwf = LwFPlugin()
  lwf_cl_strategy = Naive(lwf_model, SGD(lwf_model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), train_mb_size=100, train_epochs=1, eval_mb_size=50, evaluator=eval_plugin, plugins=[lwf])

  results = []
  for i_train, experience in enumerate(scenario.train_stream):
      res = lwf_cl_strategy.train(experience)
      results.append(lwf_cl_strategy.eval(scenario.test_stream))

  acc = results[-1]["Top1_Acc_Stream/eval_phase/test_stream/Task000"]
  forgetting = results[-1]["StreamForgetting/eval_phase/test_stream"]
  print(
      f"Final evaluation stream: {acc * 100}% accuracy, {forgetting} forgetting"
  )

  print("replay 10")
  r_10_model = get_model()
  r_10_replay = ReplayPlugin(mem_size=10)
  r_10_cl_strategy = Naive(r_10_model, SGD(r_10_model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), train_mb_size=100, train_epochs=1, eval_mb_size=50, evaluator=eval_plugin, plugins=[r_10_replay])

  results = []
  for i_train, experience in enumerate(scenario.train_stream):
      res = r_10_cl_strategy.train(experience)
      results.append(r_10_cl_strategy.eval(scenario.test_stream))

  acc = results[-1]["Top1_Acc_Stream/eval_phase/test_stream/Task000"]
  forgetting = results[-1]["StreamForgetting/eval_phase/test_stream"]
  print(
      f"Final evaluation stream: {acc * 100}% accuracy, {forgetting} forgetting"
  )

  print("replay 10 ewc")
  r_10_ewc_model = get_model()
  ewc = EWCPlugin(ewc_lambda=0.1)
  r_10_ewc_replay = ReplayPlugin(mem_size=10)
  r_10_ewc_cl_strategy = Naive(r_10_ewc_model, SGD(r_10_ewc_model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), train_mb_size=100, train_epochs=1, eval_mb_size=50, evaluator=eval_plugin, plugins=[r_10_ewc_replay, ewc])

  results = []
  for i_train, experience in enumerate(scenario.train_stream):
      res = r_10_ewc_cl_strategy.train(experience)
      results.append(r_10_ewc_cl_strategy.eval(scenario.test_stream))

  acc = results[-1]["Top1_Acc_Stream/eval_phase/test_stream/Task000"]
  forgetting = results[-1]["StreamForgetting/eval_phase/test_stream"]
  print(
      f"Final evaluation stream: {acc * 100}% accuracy, {forgetting} forgetting"
  )

  print("replay 50")
  r_50_model = get_model()
  r_50_replay = ReplayPlugin(mem_size=50)
  r_50_cl_strategy = Naive(r_50_model, SGD(r_50_model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), train_mb_size=100, train_epochs=1, eval_mb_size=50, evaluator=eval_plugin, plugins=[r_50_replay])

  results = []
  for i_train, experience in enumerate(scenario.train_stream):
      res = r_50_cl_strategy.train(experience)
      results.append(r_50_cl_strategy.eval(scenario.test_stream))

  acc = results[-1]["Top1_Acc_Stream/eval_phase/test_stream/Task000"]
  forgetting = results[-1]["StreamForgetting/eval_phase/test_stream"]
  print(
      f"Final evaluation stream: {acc * 100}% accuracy, {forgetting} forgetting"
  )

  print("replay 100")
  r_100_model = get_model()
  r_100_replay = ReplayPlugin(mem_size=100)
  r_100_cl_strategy = Naive(r_100_model, SGD(r_100_model.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(), train_mb_size=100, train_epochs=1, eval_mb_size=50, evaluator=eval_plugin, plugins=[r_100_replay])

  results = []
  for i_train, experience in enumerate(scenario.train_stream):
      res = r_100_cl_strategy.train(experience)
      results.append(r_100_cl_strategy.eval(scenario.test_stream))

  acc = results[-1]["Top1_Acc_Stream/eval_phase/test_stream/Task000"]
  forgetting = results[-1]["StreamForgetting/eval_phase/test_stream"]
  print(
      f"Final evaluation stream: {acc * 100}% accuracy, {forgetting} forgetting"
  )