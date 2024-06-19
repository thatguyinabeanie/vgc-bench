import torch


class MLP(torch.nn.Module):
    def __init__(self, n_inputs: int, hidden_layers: list[int], n_outputs: int):
        super().__init__()  # type: ignore
        layers = [torch.nn.Linear(n_inputs, hidden_layers[0]), torch.nn.ReLU()]
        for i in range(len(hidden_layers)):
            layers += [
                torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i]),
                torch.nn.ReLU(),
            ]
        layers += [torch.nn.Linear(hidden_layers[-1], n_outputs)]
        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)
