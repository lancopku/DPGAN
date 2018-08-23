# DP-GAN

This is the code used in the paper titled DP-GAN: Diversity-Promoting Generative Adversarial Network for Generating Informative and Diversified Text. The link is http://arxiv.org/abs/1802.01345


# Requirements
The software is written in tensorflow. It requires the following packages:

python3

Tensorflow 1.3

# Preparing the data

```bash
python review_generation_dataset/generate_review.py
```

The sample is shown in review_generation_dataset/train (test).
The whole Yelp dataset is avaliable at https://drive.google.com/open?id=1xCt04xWrVhbrSA7T5feV2WSukjmD4SnK

# How it works

```bash
bash run.sh
```
The default options can be edited in main.py.
 
 
# Cite

If you use this code, please cite the following paper:

@inproceedings{dp-gan,

author = {Jingjing Xu, Xu Sun, Xuancheng Ren, Junyang Lin, Binzhen Wei, Wei Li},

title = {DP-GAN: Diversity-Promoting Generative Adversarial Network for
  Generating Informative and Diversified Text},

journal = {CoRR},

volume = {abs/1802.01345},

year = {2018},

url = 
http://arxiv.org/abs/1802.01345

}


