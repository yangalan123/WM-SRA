# WM-SRA
Source code for [Weakly-Supervised Methods for Suicide Risk Assessment: Role of Related Domains (ACL 2021)](https://arxiv.org/abs/2106.02792).

Due to the ethics concern, we cannot release neither the data nor the checkpoints. Please follow the guideline as stated in [UMD Suicidality Dataset](http://users.umiacs.umd.edu/~resnik/umd_reddit_suicidality_dataset.html) to get related approvals and access for the data. 
## Citation
Please cite our paper if you find it helpful.

```
@inproceedings{yang2021weakly,
  title={Weakly-Supervised Methods for Suicide Risk Assessment: Role of Related Domains},
  author={Chenghao Yang and Yudong Zhang and Smaranda Muresan},
  booktitle={Proceedings of ACL},
  year={2021}
}
```

## Requirements
We recommend to use Anaconda to set up the environment. 
```
conda create --name <env> --file requirements.txt
```
After installing the required dependencies, you also need to download necessary data files for ``nltk`` library:
```
import nltk
nltk.download("popular")
```
## Instructions
1. After you have obtained UMD Suicidality data, extract it and move it to this project directory. The resulted directory structure may look like:
```
WM-SRA/
    umd_reddit_suicidewatch_dataset_v2/
        umd_reddit_suicidewatch_dataset_v2/
            crowd/ 
            expert/
            scripts/
    other_files_in_this_project
```

2. Move the `data_generator.py` under `umd_reddit_suicidewatch_dataset_v2/umd_reddit_suicidewatch_dataset_v2/`, then run the following line to extract necessary information from large csv files. 
```
python data_generator.py
```
3. Take a look at `config.py`. If you do not want to use pseudo-labelling, you can set `self.use_PL=False`. Otherwise, you can take a look at `data_generator.py` to see how we create pkl files for training. Then prepare your pseudo-labelling data following the same format. (Due to the ethics concern, we cannot release our pseudo-labelling data either.)
4. Simply run the following line to start training and evaluation. Our codes will do evaluation every epoch and only save the checkpoint with the best macro-F1 on the validation set.
```
python main.py --task A
```
## Acknowledgement
1. We use the pre-processing codes from [hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language)
1. We use [bert-extractive-summarizer](https://github.com/dmmiller612/bert-extractive-summarizer) to do extractive summarization over the data, which is used in multi-view learning (``K-Sum'' as in our paper).

## Some painful failure experience
1. We try to do pseudo-labelling (PL) on the fly, using the model prediction or doing random sampling to decide the labels. We even design a complicated mechanism to use these two strategies at the same time (i.e., with some probability we use the first and otherwise use the second. ) Unfortunately, all these efforts does not work.
1. We try to do contrastive learning, but it does not work. 
1. To encourage more exploration, we even implemented annealed softmax in our early versions of codes, but it does not work.
1. We try various model architectures, including adding extra attention layers and using something complicated like Transformer-RNN. But it does not work.
1. Pre-processing is important.
1. Due to the fact that this dataset is relatively small, every single point win can be significant, so you should tune the proportion of added pseudo-labelled data very very carefully. 
