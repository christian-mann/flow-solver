#!/bin/bash

ADB='adb -e'

while true; do
	$ADB exec-out screencap -p > screen.png

	python3 solve.py
	if [ $? -gt 0 ]; then
		# try resetting
		$ADB shell input tap 557 1716
	fi

	$ADB shell input tap 726 1716
	sleep 0.5
done
