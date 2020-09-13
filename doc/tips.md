
## COnvert from tf1 to tf2:
in a terminal :
> tf_upgrade_v2 --infile app.py --outfile converted/app.py


## INSTALL VENV and associate it with the project
> pip install venv
- > python3 -m venv /path/to/onlinehanzirecognizer
- > cd /path/to/onlinehanzirecognizer
- > source bin/activate

## GET THE PYTHON VERSION
import sys;
print(sys.version);

## GET TENSORFLOW VERSION :
in the python console, type in :
import tensorflow as tf;
print(tf.__version__)

#### Start tensorBoard:

>tensorboard --logdir=./log/ --port=8090 --host=127.0.0.1
In a browser , go to : http://localhost:8090/