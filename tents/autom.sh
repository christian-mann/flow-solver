#!/bin/bash

while true; do
	adb exec-out screencap -p > screen.png
	python3 TentSolver.py screen.png
	adb shell input tap 800 900
	sleep 0.5
done
