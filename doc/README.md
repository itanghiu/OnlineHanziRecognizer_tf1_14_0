## OnlineHanziRecognizer

> pip3 install --upgrade tensorflow==1.14.0
> 

#### Start tensorBoard:

>tensorboard --logdir=./log/ --port=8090 --host=127.0.0.1
In a browser , go to : http://localhost:8090/

#### Generate the training and test dataset:

> python ImageDatasetGeneration.py

#### Start training:

 Script path: PATH_TO_PROJECT\OnlineHanziRecognizer\cnn.py
 Working directory : PATH_TO_PROJECT\OnlineHanziRecognizer
> python cnn.py --mode=training

#### Launching the Web server :

- > pip install venv
- > python3 -m venv /path/to/onlinehanzirecognizer
- > cd /path/to/onlinehanzirecognizer
- > source bin/activate
- > pip install tensorflow==1.13.2
- > pip install Flask
- > pip install Image
- > pip install opencv-python
- > 

 Script path : PATH_TO_PROJECT\OnlineHanziRecognizer\app.py
 Working directory : PATH_TO_PROJECT\OnlineHanziRecognizer
> python app.py
In a browser , go to : http://localhost:5000/

 


