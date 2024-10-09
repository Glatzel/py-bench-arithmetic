Set-Location $PSScriptRoot
Set-Location ..
pixi run -e gpu pytest "./src/test_fft.py" --benchmark-autosave --benchmark-histogram=histogram/