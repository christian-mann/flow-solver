#!/bin/bash

while true; do
	adb exec-out screencap -p > screen.png

	python3 solve.py
	if [ $? -gt 0 ]; then
		# try resetting
		adb shell input tap 550 1800
	fi

	adb shell input tap 700 1800
	sleep 0.5
done
