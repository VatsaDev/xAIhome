<a target="_blank" href="https://colab.research.google.com/github/VatsaDev/xAIhome/blob/main/xai.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# XAI take home

## Making a fast videoloader

 - used any remaining parts of the sports-1m dataset from [deepvideo](https://cs.stanford.edu/people/karpathy/deepvideo/), most videos in the dataset have been lost to time, and most of the ones left are unlisted and therefore needs lots of authentication to access, I collected as many of the original videos as I could in the 4hr time period, which was 58 videos. The scrape code is in `acqi.py`
 - the 58 videos turn into 12000 4fps 5s clips, with 256x256 random cropping. in `dataload.py`, I start with the 58 clips, use ffmpeg for all the transformations, and then have a custom dataloader that makes shuffled batches and prints them out, to show the full video dataloader functionality and the time it takes
     -  I tested `dataload.py` 3 times, with 20 workers(1 core per worker), it takes ~1900-2000s to process ~12000 video clips, or **6-6.3 clips per second**
     -  speed scales around workers, 10 workers took 3000s, **3.5 clips/s**, and 2 workers takes 3-4 hrs, **0.83-1.11 clips/s**
     -  on a standard 16 core instance, this workload would be ~2400s, or **~4.5 clips/s**
 - `vaeEx.py` is my take on the "consider how to integrate it with training, e.g. overlapping data loading with training" and the VAE+Clip feature example
     - it takes data in the form `(vae_feat_1, clip_feat, vae_feat_2)` and the vae_feats are the vae features of video pairs, this is a simple NN predicting future VAE feats using current VAE feat and Clip feat
     - with the dataloader in the training process, new random crops can be made on the videos each time, better data augumentation for the neural net, and it all the files are temp in ram rather than saved/extras 
     - most of my videos were 640x360, so I made 3 256x256 crops of each one, so each one is unique but with some overlap, ensuring uniqueness and continuity of contextual information across the clips
     - should work in theory, I been haven't been able to test it due to the way way colab databuilding takes 3+ hrs, and I can't switch between cpu/gpu instances
 - `toycnn.py` is a simple cnn meant to prove that the data is stable and usable, 12669 clips with 6 classes

[colab link](https://colab.research.google.com/drive/1LRzX4N1cby0aRmvzHr1XGo6VnRsH_z0P?usp=sharing)

Mentions:

 - VAE from [HF diffusers](https://huggingface.co/docs/diffusers/v0.30.0/en/api/models/autoencoderkl)
 - clip features from: [Clip feature](https://github.com/jianjieluo/OpenAI-CLIP-Feature)
