Set-Location $PSScriptRoot
Set-Location ..
pixi run -e fft pytest "./src" --benchmark-autosave --benchmark-histogram=histogram/fft