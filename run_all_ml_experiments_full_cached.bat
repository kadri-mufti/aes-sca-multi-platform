@echo off
setlocal

cd /d "%~dp0"

echo ======================================================
echo AES-SCA Full Cached ML Runner (venv)
echo ======================================================
echo.

if not exist "venv\Scripts\activate.bat" (
  echo ERROR: venv\Scripts\activate.bat not found
  exit /b 1
)

call venv\Scripts\activate.bat
if errorlevel 1 goto :error

set ML_FAST=0

echo [1/6] STM32F4 cached models (RF/SVM/CNN)...
cd /d exp_cortexm4
python main_models_cached_stm32f4.py --hdf5 ..\analysis\traces.hdf5 --output-dir ..\results\stm32f4_ml --bytes 0 1 2 3 --cnn-epochs 25 --max-train-traces 3000 --max-test-traces 1500
if errorlevel 1 goto :error

echo [2/6] STM32F4 legacy comparison refresh...
python compare_all_methods_stm32f4.py --results-dir ..\results\stm32f4_ml --output comparison_stm32f4.txt
if errorlevel 1 goto :error

echo [3/6] STM32F4 pretty comparison...
python compare_all_methods_stm32f4_pretty.py --results-dir ..\results\stm32f4_ml --output-md comparison_stm32f4_pretty.md --output-csv comparison_stm32f4_pretty.csv
if errorlevel 1 goto :error

echo [4/6] AES-HD cached models (RF/SVM/CNN)...
cd /d ..\exp_aeshd_hd
python main_models_cached_aeshd.py --dataset ..\analysis\AES_HD_dataset --output-dir ..\results\aeshd_ml --cnn-epochs 25 --max-prof-traces 10000 --max-attack-traces 4000
if errorlevel 1 goto :error

echo [5/6] AES-HD legacy comparison refresh...
python compare_all_methods_aeshd.py --results-dir ..\results\aeshd_ml --output comparison_aeshd.txt
if errorlevel 1 goto :error

echo [6/6] AES-HD pretty comparison...
python compare_all_methods_aeshd_pretty.py --results-dir ..\results\aeshd_ml --output-md comparison_aeshd_pretty.md --output-csv comparison_aeshd_pretty.csv
if errorlevel 1 goto :error

cd /d ..
echo.
echo ======================================================
echo Full cached ML pipeline completed successfully.
echo Results and artifacts are under results\stm32f4_ml and results\aeshd_ml
echo ======================================================
goto :eof

:error
echo.
echo ======================================================
echo ERROR: Full cached ML pipeline stopped due to a failure.
echo Check output above for the failing step.
echo ======================================================
exit /b 1
