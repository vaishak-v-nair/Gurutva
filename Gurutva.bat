@echo off
REM Double-click this to open Gurutva.
cd /d "%~dp0"
py -3 -m src.product.app
if errorlevel 1 pause
