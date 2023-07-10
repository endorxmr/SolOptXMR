#!/bin/bash -e

cd externals/tsqsim/
./util/deps-pull.sh
cd ../..

if [ "$(uname)" == "Darwin" ]; then
	HOMEBREW_NO_AUTO_UPDATE=1 brew install rapidjson libffi osx-cpu-temp
	echo "Please enable the `at` scheduler manually:"
	echo "https://unix.stackexchange.com/questions/478823/making-at-work-on-macos/478840#478840"
elif [ "$(uname)" == "Linux" ]; then
	sudo apt install gfortran libffi-dev python3-testresources
	#sudo apt install libboost-all-dev # Only JSON is needed for now
	#sudo apt install libffi-dev
	sudo apt install rapidjson-dev lm-sensors fswebcam at
fi

if pip3 install -r requirements.txt ; then
	echo "pip succeeded."
else
	echo "pip failed. Trying a fallback."
	pip3 uninstall -r requirements.txt -y
	pip3 install pandas>=2.0.3 scipy>=1.11.1 numpy>=1.25.1 matplotlib \
	pvlib pykrakenapi pyparsing pyrsistent python-dateutil \
	python-json-config pytz requests six urllib3 \
	beautifulsoup4 wget cairosvg Pillow>=9.2.0 geocoder \
	opencv-python screeninfo ortools \
	# pytesseract not employed yet
	#pip3 install --upgrade 
	# Curses menu:
	#windows-curses==2.3.0 ; sys_platform == 'win32'
	# Ascii menu:
	#asciimatics==1.13.0
fi

