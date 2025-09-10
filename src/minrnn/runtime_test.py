from statistics import mean, stdev
from timeit import default_timer as timer

import torch
from torch import Tensor, nn

from .minrnn_cell import MinGRUCell, MinLSTMCell, MinRNNCellBase


def print_info(message: str) -> None:
    print("\033[92m" + "[INFO] " + message + "\033[0m")


def print_runtime_result(lap_times: list[float]) -> None:
    num_repeats = len(lap_times)
    duration = sum(lap_times)
    average = mean(lap_times)
    std = stdev(lap_times)

    print(f"Duration: {duration:.2f} [s]")
    print(
        f"{average:.4f} [s] " "\u00b1" f" {std:.4f} [s] of {num_repeats} runs"
    )


def rnn_forward_backward(rnn: nn.RNNCellBase, inputs: Tensor) -> None:
    """Forward-backward process of conventional RNNCells

    Args:
        - rnn: torch.nn.RNNCellBase. The conventional PyTorch RNNCells
        - inputs: torch.Tensor. Sequential inputs of shape (N, L, H_in).
    """
    seq_length = inputs.size(1)

    # forward process
    state: Tensor | tuple[Tensor, Tensor] = None
    l2: Tensor = 0.0
    for seq_i in range(seq_length):
        state = rnn(inputs[:, seq_i], state)

        # accumulate L2 norm for the backward process
        if isinstance(state, tuple):  # LSTMCell
            h = state[0]
        else:  # GRUCell
            h = state
        l2 = l2 + torch.mean(h.norm(dim=-1))

    # backward process
    l2.backward()


def minrnn_forward_backward(rnn: MinRNNCellBase, inputs: Tensor) -> None:
    """Forward-backward process of MinRNNs

    Args:
        - rnn: torch.nn.RNNCellBase. The conventional PyTorch RNNCells
        - inputs: torch.Tensor. Sequential inputs of shape (N, L, H_in).
    """
    # forward process
    pred_states: Tensor = rnn(inputs, hx=None)  # (N, L, H_hid)

    # accumulate L2 norm for the backward process
    l2: Tensor = torch.sum(torch.mean(pred_states.norm(dim=-1), dim=-1))

    # backward process
    l2.backward()


def runtime_test(
    batch_size: int = 64,
    seq_length: int = 512,
    input_size: int = 64,
    alpha: float = 2.0,
    num_repeats: int = 100,
    device: torch.DeviceObjType = torch.device("cpu"),
) -> None:
    """Compare runtimes of conventional RNNs and MinRNN. The official
    experiment is shown in Section 4.1 'Runtime.' and Fig. 1 left.

    Args:
        - batch_size: int. Batch size of the dummy sequential input.
        - seq_length: int. Length of the sequence.
        - input_size: int. Input size.
        - alpha: float. Expansion parameter of the number of hidden units.
        - num_repeats: int. The number of runs of each computation.
        - device: torch.DeviceObjType. Device for computation.
    """
    print_info("Start runtime_test")
    print_info(f"Compute on {str(device).upper()}")

    hidden_size = int(input_size * alpha)
    print_info("Display parameters")
    print(f"- batch_size = {batch_size}")
    print(f"- seq_length = {seq_length}")
    print(f"- input_size = {input_size}")
    print(f"- alpha = {alpha}")
    print(f"- hidden_size = {hidden_size}")

    # prepare RNN cells
    gru = nn.GRUCell(input_size, hidden_size).to(device)
    lstm = nn.LSTMCell(input_size, hidden_size).to(device)
    min_gru = MinGRUCell(input_size, hidden_size).to(device)
    min_lstm = MinLSTMCell(input_size, hidden_size).to(device)

    # define an input tensor
    dummy_inputs = torch.randn(
        (batch_size, seq_length, input_size), device=device
    )

    print_info(f"Run the forward-backward processes for {num_repeats} times")

    # conventional RNNCells
    for cell in (gru, lstm):
        name = type(cell).__name__
        print_info(f"Start {name}")

        lap_times = [0.0] * num_repeats
        for rep_i in range(num_repeats):
            lap_start = timer()
            print(f"Running {rep_i + 1} / {num_repeats}\r", end="")

            rnn_forward_backward(cell, dummy_inputs)

            lap_times[rep_i] = timer() - lap_start

        print_info(f"Finish {name}")
        print_runtime_result(lap_times)

    # MinRNNs
    for cell in (min_gru, min_lstm):
        name = type(cell).__name__
        print_info(f"Start {name}")

        lap_times = [0.0] * num_repeats
        for rep_i in range(num_repeats):
            lap_start = timer()
            print(f"Running {rep_i + 1} / {num_repeats}\r", end="")

            minrnn_forward_backward(cell, dummy_inputs)

            lap_times[rep_i] = timer() - lap_start

        print_info(f"Finish {name}")
        print_runtime_result(lap_times)


def runtime_test_main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser("Parameters of runtime_test")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-s", "--seq_length", type=int, default=512)
    parser.add_argument("-i", "--input_size", type=int, default=64)
    parser.add_argument("-a", "--alpha", type=float, default=2.0)
    parser.add_argument("-r", "--num_repeats", type=int, default=100)
    parser.add_argument(
        "-d", "--device", type=str, default="cpu", choices=["cpu", "cuda"]
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    runtime_test(
        args.batch_size,
        args.seq_length,
        args.input_size,
        args.alpha,
        args.num_repeats,
        device,
    )


if __name__ == "__main__":
    runtime_test_main()
