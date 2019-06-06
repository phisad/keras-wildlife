#!/bin/sh
git pull
python3 setup.py install --prefix='/home/users/sadler/.local'
python3 setup.py clean -a
