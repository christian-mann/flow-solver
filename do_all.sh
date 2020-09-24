#!/bin/bash

while true; do
	adb exec-out screencap -p > screen.png
	sleep 1

	python3 solve.py
	sleep 1

	adb shell input tap 700 1800
	sleep 1
done
