# train
python src/train.py -i=src/train_data/train.csv -o=outputs -a=0 -p=1 -t=2
# evaluate, lastep means the last epoch
python src/predict.py -i=src/test_data/test.csv -m=outputs/lastep -a=0 -p=1
