# XAI take home

## Making a fast videoloader

Basing this similar in purpose to a pytorch dataloader, loads a full dataset for the model to train on

Step 1: accquiring the entire sports-1m dataset 

 - datas old, only parts of it are left on yt, and the only reason they still remain is due to being unlisted, therefore requiring account auth, making them inaccessible, so very very few remain, have to start with salvaging the dataset

 - dataload.py, first full dataset making and batching script, can make all the random cropped clip pairs and make batches of all 12000 vids in 1900-2000 seconds, or about ~6-6.3 videos per second
 - running with 10 workers takes 3000 seconds, ~3.5 videos per second, near linear scaling with cores
 - this workload would be ~40min on a standard 16 core cpu, or ~4.5 videos per second

Mentions:

 - clip features from: https://github.com/jianjieluo/OpenAI-CLIP-Feature
