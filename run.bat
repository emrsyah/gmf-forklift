@echo off
rem Launcher Forklift Safety AI (laptop, Fase 1).
rem   Klik dua kali, atau dari terminal:  run            (menu)
rem                                       run live       (langsung tes live)
rem Pilihan langsung: live, live-noyolo, pose, record, diag, replay, test, phone, front
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [!] .venv belum ada. Jalankan dulu:  uv venv --python 3.10 .venv  lalu  uv pip install -r requirements.txt
  pause
  exit /b 1
)

set "CHOICE=%~1"
if not "%CHOICE%"=="" goto :run

:menu
echo.
echo  ==============================================
echo   Forklift Safety AI - laptop
echo  ==============================================
echo   1. Tes live webcam (lengkap, dengan YOLO)
echo   2. Tes live webcam tanpa YOLO (wajah saja, tanpa D-06/D-10)
echo   3. Cek pose kepala (30 detik)
echo   4. Rekam skenario Lembar A (manual, hotkey)
echo   5. Rekam diagnostik terpandu (mata, lensa, pangkuan)
echo   6. Replay harness (laporan dari eval\scenarios\truth.csv)
echo   7. Jalankan semua unit test
echo   8. Tes live kamera HP + profil kamera (simulasi kamera Jetson)
echo   9. Kamera depan (webcam / HP, opsional profil kamera)
echo   0. Keluar
echo.
echo   Di jendela kamera: q keluar, c kalibrasi ulang, d panel debug
echo.
set /p "CHOICE=Pilih [1]: "
if "%CHOICE%"=="" set "CHOICE=1"

:run
if /i "%CHOICE%"=="1"           goto :live
if /i "%CHOICE%"=="live"        goto :live
if /i "%CHOICE%"=="2"           goto :noyolo
if /i "%CHOICE%"=="live-noyolo" goto :noyolo
if /i "%CHOICE%"=="3"           goto :pose
if /i "%CHOICE%"=="pose"        goto :pose
if /i "%CHOICE%"=="4"           goto :record
if /i "%CHOICE%"=="record"      goto :record
if /i "%CHOICE%"=="5"           goto :diag
if /i "%CHOICE%"=="diag"        goto :diag
if /i "%CHOICE%"=="6"           goto :replay
if /i "%CHOICE%"=="replay"      goto :replay
if /i "%CHOICE%"=="7"           goto :test
if /i "%CHOICE%"=="test"        goto :test
if /i "%CHOICE%"=="8"           goto :phone
if /i "%CHOICE%"=="phone"       goto :phone
if /i "%CHOICE%"=="9"           goto :front
if /i "%CHOICE%"=="front"       goto :front
if /i "%CHOICE%"=="0"           exit /b 0
echo Pilihan tidak dikenal: %CHOICE%
set "CHOICE="
goto :menu

:live
"%PY%" -m src.main --source webcam
goto :done

:noyolo
"%PY%" -m src.main --source webcam --no-yolo
goto :done

:pose
"%PY%" -m tools.check_pose
goto :done

:record
set /p "PERAGA=ID peraga (mis. P1, jangan nama lengkap): "
set /p "CAHAYA=Cahaya [normal/redup, default normal]: "
if "%CAHAYA%"=="" set "CAHAYA=normal"
"%PY%" -m tools.recorder --peraga %PERAGA% --cahaya %CAHAYA%
goto :done

:diag
set /p "PERAGA=ID peraga (mis. P1): "
"%PY%" -m tools.recorder --peraga %PERAGA% --session diag_%PERAGA%_%RANDOM% --truth eval\scenarios\diag_truth.csv --guide configs\guides\diag_eyes_cover_lap.yaml
goto :done

:replay
"%PY%" -m eval.replay_eval
goto :done

:test
"%PY%" -m pytest -q
goto :done

:phone
echo.
echo   Android: aplikasi "IP Webcam" -^> Start server -^> URL http://IP-HP:8080/video
echo   iPhone / DroidCam / Iriun sebagai webcam: isi angka webcam, mis. 1
echo   Panduan lengkap: docs\PHONE_CAMERA_TEST.md
set "SRC="
set /p "SRC=URL stream HP atau angka webcam: "
if "%SRC%"=="" (
  set "CHOICE="
  goto :menu
)
echo.
"%PY%" -m src.capture.camera_sim --list --role kabin
set "PROFIL="
set /p "PROFIL=Profil kamera [imx219_77ir]: "
if "%PROFIL%"=="" set "PROFIL=imx219_77ir"
set "HFOV="
set /p "HFOV=HFOV kamera HP (derajat) [kosong = nilai di camera_profiles.yaml]: "
set "HFOVARG="
if not "%HFOV%"=="" set "HFOVARG=--phone-hfov %HFOV%"
"%PY%" -m src.main --source "%SRC%" --camera-profile %PROFIL% %HFOVARG%
goto :done

:front
echo.
echo   Sumber: angka webcam (mis. 1), URL HP (http://IP-HP:8080/video), atau path video
set "SRC="
set /p "SRC=Sumber kamera depan [1]: "
if "%SRC%"=="" set "SRC=1"
echo.
"%PY%" -m src.capture.camera_sim --list --role depan
set "PROFIL="
set /p "PROFIL=Profil kamera [none = tanpa simulasi]: "
if "%PROFIL%"=="" set "PROFIL=none"
set "HFOV="
set /p "HFOV=HFOV kamera HP (derajat) [kosong = nilai di camera_profiles.yaml]: "
set "HFOVARG="
if not "%HFOV%"=="" set "HFOVARG=--phone-hfov %HFOV%"
"%PY%" tools\run_front.py --source "%SRC%" --camera-profile %PROFIL% %HFOVARG%
goto :done

:done
echo.
pause
endlocal
