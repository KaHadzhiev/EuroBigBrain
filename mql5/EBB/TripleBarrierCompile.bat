@echo off
REM EBB_TripleBarrier compile harness.
REM Usage: TripleBarrierCompile.bat

setlocal
set ME="C:\MT5-Instances\Instance1\MetaEditor64.exe"
set SRC="C:\Users\kahad\IdeaProjects\EuroBigBrain\mql5\EBB_TripleBarrier.mq5"
set LOG="C:\Users\kahad\IdeaProjects\EuroBigBrain\mql5\EBB_TripleBarrier.log"

if not exist %ME% (
    echo FAIL: MetaEditor not found at %ME%
    exit /b 2
)

echo Compiling EBB_TripleBarrier.mq5 ...
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
