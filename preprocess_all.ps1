conda activate cv_ss23

ECHO y | python ./preprocess.py --batch=batch_20230912

ECHO n | python ./preprocess.py --batch=batch_20230919

ECHO n | python ./preprocess.py --batch=batch_202301027