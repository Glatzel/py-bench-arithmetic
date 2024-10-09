Set-Location $PSScriptRoot
Set-Location ..
pixi run -e gpu pytest "./src/test_broadcast1.py" --benchmark-autosave --benchmark-histogram=histogram/