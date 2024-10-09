Set-Location $PSScriptRoot
Set-Location ..
pixi run -e gpu pytest "./src/test_datum_compense_loop.py" --benchmark-autosave --benchmark-histogram=histogram/