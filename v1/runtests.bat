REM Embarrassing that this is a Windows batch file not sensible Linux shell script
REM but for some reason my git bash doesn't run python.
REM So this is a quick workaround.
REM

@echo off
python TestMakeFrameNumeric.py
python TestTrainPredict.py
python TestCalcMeanStdPredictions.py
python TestSpotErrors.py
