Set-Location $PSScriptRoot
Set-Location ..
pixi run -e gpu pytest "./src" --benchmark-autosave --benchmark-histogram=histogram/fft