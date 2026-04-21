@echo off
REM EBB_Core compile harness - W1.1 skeleton verification
REM Per memory reference_metaeditor_compile.md

setlocal
set ME="C:\MT5-Instances\Instance1\MetaEditor64.exe"
set SRC="C:\Users\kahad\IdeaProjects\EuroBigBrain\mql5\EBB_Core.mq5"
set LOG="C:\Users\kahad\IdeaProjects\EuroBigBrain\mql5\EBB_Core.log"

if not exist %ME% (
    echo FAIL: MetaEditor not found at %ME%
    exit /b 2
)

echo Compiling EBB_Core.mq5 ...
%ME% /compile:%SRC% /log:%LOG%
REM MetaEditor returns 0 regardless; parse log for errors.

if not exist %LOG% (
    echo FAIL: no log produced at %LOG%
    exit /b 3
)

findstr /C:"0 error" %LOG% >nul
if errorlevel 1 (
    echo FAIL: errors found in %LOG%
    type %LOG%
    exit /b 1
)

echo PASS: 0 errors
type %LOG%
exit /b 0
