#!/bin/bash

while true; do
	adb exec-out screencap -p > screen.png

	python3 solve.py

	adb shell input tap 700 1800
	sleep 1
done
