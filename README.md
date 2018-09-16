
### README

### TODO

labelCharMapFile is duplicated
checkpoint directory relative reference

### Starting the server :

> python webApp/app.py


### Server url :

http://localhost:5000/

## Training :

> python chinese_character_recognition_bn.py --mode=train --max_steps=16002 --eval_steps=100 --save_steps=500

## Validating :

> python chinese_character_recognition_bn.py --mode=validation

## Inferring :

> python chinese_character_recognition_bn.py --mode=inference