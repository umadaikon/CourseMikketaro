@echo off
set local
set baseDir=C:\openpose
set imgDir=C:\openpose\wallimg\%1
set csvDir=C:\openpose\holdposition\%1.csv
:set height=%2

cd C:\openpose\getHold\hold_detection

python hold_slice_using2.py %imgDir%

cd /d %~dp0