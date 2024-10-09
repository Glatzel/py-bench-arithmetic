Set-Location $PSScriptRoot
Set-Location ..
pixi run -e gpu pytest "./src/test_broadcast2.py" --benchmark-autosave --benchmark-histogram=histogram/