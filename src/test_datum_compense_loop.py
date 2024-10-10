import os

import numpy as np
import pytest

group = "datum compense loop "


def data_figures():
    if os.getenv("CI"):
        return [0]
    else:  # pragma: nocover
        return range(6)


@pytest.fixture(params=data_figures(), scope="module")
def sample_coords(request):
    return 10**request.param, 2821940.796, 469704.6693, 400.0


def foo(x1, y1, h1):
    q = h1 / 6378_137
    factor = q / (1 + q)
    x1 = x1 - factor * (x1 - 500000)
    y1 = y1 - factor * (y1 - 0)


def test_python(benchmark, sample_coords):
    def loop(n, x, y, h):
        for _ in range(n):
            foo(x, y, h)

    benchmark.group = group + str(sample_coords[0])
    benchmark(loop, *sample_coords)


def test_numba(benchmark, sample_coords):
    numba = pytest.importorskip("numba")

    @numba.njit()
    def foo_numba(x1, y1, h1, r=6378_137.0, x0=0.0, y0=500000.0):  # pragma: nocover
        q = h1 / r
        factor = q / (1 + q)
        x1 = x1 - factor * (x1 - y0)
        y1 = y1 - factor * (y1 - x0)

        return x1, y1

    def loop(n, x, y, h):
        for _ in range(n):
            foo_numba(x, y, h)

    benchmark.group = group + str(sample_coords[0])
    benchmark(loop, *sample_coords)


def test_asarray(benchmark, sample_coords):
    def loop(n, x, y, h):
        for _ in range(n):
            q = h / 6378_137
            factor = q / (1 + q)
            x1 = np.asarray(x, np.float64)
            y1 = np.asarray(y, np.float64)
            x1 = x1 - factor * (x1 - 500000)
            y1 = y1 - factor * (y1 - 0)

    benchmark.group = group + str(sample_coords[0])
    benchmark(loop, *sample_coords)


def test_tensor(benchmark, sample_coords):
    torch = pytest.importorskip("torch")

    def loop(n, x, y, h):
        for _ in range(n):
            q = h / 6378_137
            factor = q / (1 + q)
            x1 = torch.as_tensor(x, dtype=torch.float64)
            y1 = torch.as_tensor(y, dtype=torch.float64)
            x1 = x1 - factor * (x1 - 500000)
            y1 = y1 - factor * (y1 - 0)

    benchmark.group = group + str(sample_coords[0])
    benchmark(loop, *sample_coords)
