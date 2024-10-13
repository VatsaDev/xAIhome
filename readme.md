# XAI take home

## Making a fast videoloader

 - used any remaining parts of the sports-1m dataset from [deepvideo](https://cs.stanford.edu/people/karpathy/deepvideo/), most videos in the dataset have been lost to time, and most of the ones left are unlisted and therefore needs lots of authentication to access, I collected as many of the original videos as I could in the 4hr time period, which was 58 videos. The scrape code is in `acqi.py`
 - the 58 videos turn into 12000 4fps 5s clips, with 256x256 random cropping. in `dataload.py`, I start with the 58 clips, use ffmpeg for all the transformations, and then have a custom dataloader that makes shuffled batches and prints them out, to show the full video dataloader functionality and the time it takes
     -  I tested `dataload.py` 3 times, with 20 workers(1 core per worker), it takes ~1900-2000s to process ~12000 video clips, or 6-6.3 clips per second
     -  speed scales around workers, 10 workers took 3000s, 3.5 clips/s, and 2 workers takes 3-4 hrs, 0.83-1.11 clips/s
     -  on a standard 16 core instance, this workload would be ~2400s, or ~4.5 videos/s

Mentions:

 - VAE from [HF diffusers](https://huggingface.co/docs/diffusers/v0.30.0/en/api/models/autoencoderkl)
 - clip features from: [Clip feature](https://github.com/jianjieluo/OpenAI-CLIP-Feature)
