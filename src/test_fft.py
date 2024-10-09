import os

import numpy as np
import pyfftw
import pytest
import scipy.fft as sp_fft
import ssqueezepy
import torch
import torch.fft as t_fft

group = "FFT size: "

rng = np.random.default_rng(1337)


def data_figures():
    if os.getenv("CI"):
        return [10]
    else:  # pragma: nocover
        return range(10, 27, 2)


@pytest.fixture(params=data_figures(), scope="module")
def data_size(request):
    return request.param


def test_numpy(benchmark, data_size):
    def foo(data):
        np.fft.fft(data)

    size = 2**data_size
    data = rng.normal(0, 1, size=size) + 1j * rng.normal(0, 1, size=size)
    benchmark.group = group + f"2^{data_size}"
    benchmark.name = "numpy"
    benchmark(foo, data)


def test_scipy(benchmark, data_size):
    def foo(data):
        sp_fft.fft(data)

    size = 2**data_size
    data = rng.normal(0, 1, size=size) + 1j * rng.normal(0, 1, size=size)
    benchmark.group = group + f"2^{data_size}"
    benchmark.name = "scipy"
    benchmark(foo, data)


def test_fftw(benchmark, data_size):
    def foo(data):
        pyfftw.interfaces.numpy_fft.fft(data)

    size = 2**data_size
    data = rng.normal(0, 1, size=size) + 1j * rng.normal(0, 1, size=size)
    benchmark.group = group + f"2^{data_size}"
    benchmark.name = "pyfftw"
    benchmark(foo, data)


def test_torch(benchmark, data_size):
    def foo(data):
        t_fft.fft(data)

    size = 2**data_size
    data = torch.normal(0.0, 1.0, size=(size,), dtype=torch.cfloat)
    benchmark.name = "torch"
    benchmark.group = group + f"2^{data_size}"
    benchmark(foo, data)


def test_ssqueezepy(benchmark, data_size):
    def foo(data):
        ssqueezepy.fft(data)

    os.environ["SSQ_PARALLEL"] = "1"
    size = 2**data_size
    data = rng.normal(0, 1, size=size) + 1j * rng.normal(0, 1, size=size)
    benchmark.group = group + f"2^{data_size}"
    benchmark.name = "ssqueezepy"
    benchmark(foo, data)
