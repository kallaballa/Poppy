#!/bin/sh
export DOCUMENT_ROOT=/home/elchaschab/devel/Poppy/src
export SCRIPT_NAME=./get.php
export SCRIPT_FILENAME=/home/elchaschab/devel/Poppy/src/get.php
export REDIRECT_STATUS=200
exec /usr/bin/php-cgi /home/elchaschab/devel/Poppy/src/get.php res=poppy.html
