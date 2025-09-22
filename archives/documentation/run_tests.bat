@echo off
echo Activating conda environment ift3150...
call conda activate ift3150

echo.
echo Running basic structure tests...
python test_simple.py

echo.
echo Running API tests (this will make real NHL API calls)...
python test_api.py

echo.
echo Tests complete! Check results above.
pause