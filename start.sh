echo 'start training train_changevocab'

cd /home/chan/bpu/python/cnn-text-classification-tf 

pwd

sed -i -e 's/raw_sentiment_train/raw_test_train_md30_mf10/' multi_class_data_loader.py
#sed -i -e 's/min_frequency = 10/min_frequency = 10/' word_data_processor.py
python3 train_changevocab.py

j=10
for i in {20..80..10}
do
prev=`expr $i - $j`
sed -i -e 's/raw_test_train_md30_mf'$prev'/raw_test_train_md30_mf'$i'/' multi_class_data_loader.py
sed -i -e 's/min_frequency = '$prev'/min_frequency = '$i'/' word_data_processor.py
python3 train_changevocab.py
done

#python3 train_changevocab.py

sed -i -e 's/raw_test_train_md30_mf80/raw_test_train_md30_mf100/' multi_class_data_loader.py
sed -i -e 's/min_frequency = 80/min_frequency = 100/' word_data_processor.py
python3 train_changevocab.py
sed -i -e 's/raw_test_train_md30_mf100/raw_test_train_md30_mf200/' multi_class_data_loader.py
sed -i -e 's/min_frequency = 100/min_frequency = 200/' word_data_processor.py
python3 train_changevocab.py

sed -i -e 's/raw_test_train_md30_mf200/raw_sentiment_train/' multi_class_data_loader.py
sed -i -e 's/min_frequency = 200/min_frequency = 10/' word_data_processor.py

