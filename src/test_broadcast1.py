import os

import numpy as np
import pytest
import torch


def data_figures():
    if os.getenv("CI"):
        return [0]
    else:  # pragma: nocover
        return range(9)


@pytest.fixture(params=data_figures(), scope="module")
def sample_data(request):
    n = 10**request.param
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    z = np.empty(n, dtype=np.float64)
    return request.param, x, y, z


group = "Round1 "


def test_np1(benchmark, sample_data):
    def foo(x, y, z):
        z = 2 * y + 4 * x  # noqa: F841

    benchmark.group = group + f"10^{sample_data[0]}"
    benchmark.name = "numpy"
    benchmark(foo, *sample_data[1:])


def test_ne1(benchmark, sample_data):
    ne = pytest.importorskip("numexpr")

    def foo(x, y, z):
        ne.evaluate("2*y + 4*x", out=z)

    benchmark.group = group + f"10^{sample_data[0]}"
    benchmark.name = "numexpr"
    benchmark(foo, *sample_data[1:])


def test_torch1(benchmark, sample_data):
    def foo(x, y, z):
        z = 2 * y + 4 * x  # noqa: F841

    benchmark.name = "torch"

    benchmark.group = group + f"10^{sample_data[0]}"
    benchmark(
        foo,
        torch.from_numpy(sample_data[1]),
        torch.from_numpy(sample_data[2]),
        torch.from_numpy(sample_data[3]),
    )


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="No cuda device.")
def test_torch_cuda1(benchmark, sample_data):  # pragma: nocover
    def foo(x, y, z):
        z = 2 * y + 4 * x  # noqa: F841

    benchmark.group = group + f"10^{sample_data[0]}"
    benchmark.name = "torch cuda"
    benchmark(
        foo,
        torch.from_numpy(sample_data[1]).cuda(),
        torch.from_numpy(sample_data[2]).cuda(),
        torch.from_numpy(sample_data[3]).cuda(),
    )
