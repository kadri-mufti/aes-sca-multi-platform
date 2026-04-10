@echo off
setlocal

REM Run from repository root (directory of this .bat)
cd /d "%~dp0"

echo ======================================================
echo AES-SCA ML Pipeline Runner
echo ======================================================
echo.

set ML_FAST=1

@REM REM Optional dependency install (set SKIP_INSTALL=1 to skip)
@REM if "%SKIP_INSTALL%"=="1" goto :skip_install
@REM echo [1/9] Installing Python dependencies...
@REM python -m pip install -r requirements.txt
@REM if errorlevel 1 goto :error
@REM :skip_install

echo.
echo [2/9] Running STM32F4 Random Forest...
cd /d exp_cortexm4
python main_rf_stm32f4.py --hdf5 ..\analysis\traces.hdf5 --output-dir ..\results\stm32f4_ml
if errorlevel 1 goto :error

echo.
echo [3/9] Running STM32F4 SVM...
python main_svm_stm32f4.py --hdf5 ..\analysis\traces.hdf5 --kernel linear --output-dir ..\results\stm32f4_ml
if errorlevel 1 goto :error

echo.
echo [4/9] Running STM32F4 1D CNN...
python main_cnn1d_stm32f4.py --hdf5 ..\analysis\traces.hdf5 --epochs 5 --output-dir ..\results\stm32f4_ml
if errorlevel 1 goto :error

echo.
echo [5/9] Running STM32F4 feature-selection experiments...
python main_rf_pca_stm32f4.py --hdf5 ..\analysis\traces.hdf5 --n-bytes 2 --output-dir ..\results\stm32f4_ml
if errorlevel 1 goto :error

echo.
echo [6/9] Running STM32F4 comparison...
python compare_all_methods_stm32f4.py --results-dir ..\results\stm32f4_ml --output comparison_stm32f4.txt
if errorlevel 1 goto :error

echo.
echo [7/9] Running AES-HD Random Forest...
cd /d ..\exp_aeshd_hd
python main_rf_aeshd.py --dataset ..\analysis\AES_HD_dataset --output-dir ..\results\aeshd_ml
if errorlevel 1 goto :error

echo.
echo [8/9] Running AES-HD SVM...
python main_svm_aeshd.py --dataset ..\analysis\AES_HD_dataset --kernel linear --output-dir ..\results\aeshd_ml
if errorlevel 1 goto :error

echo.
echo [9/9] Running AES-HD 1D CNN and comparison...
python main_cnn1d_aeshd.py --dataset ..\analysis\AES_HD_dataset --epochs 5 --output-dir ..\results\aeshd_ml
if errorlevel 1 goto :error
python compare_all_methods_aeshd.py --results-dir ..\results\aeshd_ml --output comparison_aeshd.txt
if errorlevel 1 goto :error

echo.
echo ======================================================
echo All ML experiments completed successfully.
echo Results saved to:
echo   results\stm32f4_ml
echo   results\aeshd_ml
echo ======================================================
goto :eof

:error
echo.
echo ======================================================
echo ERROR: Pipeline stopped because a command failed.
echo Check the output above to identify the failing step.
echo ======================================================
exit /b 1
