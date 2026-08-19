@echo off
REM ============================================================
REM Remote-Friendly Setup Script — Windows
REM ============================================================
REM Usage:
REM   git clone https://github.com/seealake/Textual-Rule-Explanations-of-Reinforcement-Learning.git
REM   cd Textual-Rule-Explanations-of-Reinforcement-Learning
REM   setup.bat
REM ============================================================

setlocal enabledelayedexpansion

echo ==========================================
echo   Project Setup (Windows/Remote)
echo ==========================================

echo [1/4] Creating Python virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat
echo   Virtual environment created at .venv\

echo [2/4] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

echo [3/4] Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt

echo Detecting GPU runtime for PyTorch...
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
	if "%SKIP_CUDA_TORCH%"=="1" (
		echo   SKIP_CUDA_TORCH=1 set. Installing default torch wheels.
		python -m pip install --upgrade torch torchvision torchaudio
	) else (
		if "%TORCH_CUDA_WHL_INDEX%"=="" set TORCH_CUDA_WHL_INDEX=https://download.pytorch.org/whl/cu121
		echo   NVIDIA GPU detected. Trying CUDA torch from %TORCH_CUDA_WHL_INDEX%
		python -m pip install --upgrade torch torchvision torchaudio --index-url %TORCH_CUDA_WHL_INDEX%
		if errorlevel 1 (
			echo   CUDA torch install failed. Falling back to default PyPI wheels.
			python -m pip install --upgrade torch torchvision torchaudio
		)
	)
) else (
	echo   NVIDIA GPU not detected. Installing default torch wheels.
	python -m pip install --upgrade torch torchvision torchaudio
)

echo [4/4] Verifying installation...
python -c "import gymnasium as gym; import stable_baselines3; import sklearn, numpy, pandas, matplotlib, seaborn, kneed, yaml, pygame, minigrid, torch; import highway_env; print('  gymnasium ' + gym.__version__); print('  stable-baselines3 ' + stable_baselines3.__version__); print('  scikit-learn ' + sklearn.__version__); print('  numpy ' + numpy.__version__); print('  pandas ' + pandas.__version__); print('  matplotlib ' + matplotlib.__version__); print('  seaborn ' + seaborn.__version__); print(f'  torch {torch.__version__} (cuda={torch.version.cuda}, available={torch.cuda.is_available()})'); env=gym.make('LunarLander-v3'); env.reset(seed=0); env.close(); print('  LunarLander-v3 OK'); env=gym.make('MiniGrid-Dynamic-Obstacles-8x8-v0'); env.reset(seed=0); env.close(); print('  MiniGrid-Dynamic-Obstacles-8x8-v0 OK'); env=gym.make('merge-v0'); env.reset(seed=0); env.close(); print('  merge-v0 OK'); env=gym.make('intersection-v0'); env.reset(seed=0); env.close(); print('  intersection-v0 OK'); print('  All dependencies verified.')"
python -m pip check

echo.
echo Testing config loader...
python -m experiments.config_loader

echo.
echo ==========================================
echo   Setup complete!
echo   Activate env:  .venv\Scripts\activate
echo ==========================================

endlocal
