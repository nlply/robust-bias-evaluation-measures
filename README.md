# Robust Evaluation Measures for Evaluating Social Biases in Masked Language Models
> Code for the paper ***Robust Evaluation Measures for Evaluating Social Biases in Masked Language Models*** (accepted by **AAAI 2024**)

## 🚴Prepare Model
Download the model from [huggingface](https://huggingface.co/) with the following script:


```
mkdir models
cd models/
git clone https://huggingface.co/albert-base-v2
git clone https://huggingface.co/bert-base-cased
git clone https://huggingface.co/roberta-large
```

## 💻Prepare Datasets
Download [CrowS-Pairs (CP)](https://github.com/nyu-mll/crows-pairs) and [StereoSet (SS)](https://github.com/moinnadeem/StereoSet) datasets using the following script:
```
mkdir data
wget -O data/cp.csv https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv
wget -O data/ss.json https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json
```


## 🧘Preprocessing

The original data is already in the `data` folder, if not, please download it in [CrowS-Pairs (CP)](https://github.com/nyu-mll/crows-pairs) and [StereoSet (SS)](https://github.com/moinnadeem/StereoSet)

Then, preprocess the data with the following script:
```
cd code/
python preprocessing.py --input stereoset --output ../data/paralled_ss.json
python preprocessing.py --input crows_pairs --output ../data/paralled_cp.json
```
> We refer to the method of [Kaneko et al.](https://github.com/kanekomasahiro/evaluate_bias_in_mlm) to preprocess the data

## 💇‍♂️Data Sampling
Use the following script to sample the data, the sampling ratio is 30%, 40%, 50%, 60%, 70% and 80%:
```
cd code/
python sampling.py --sample_rate [sample_rate]
```
You can set `[sample_rate]` to 0.8 for 80% sampling.

## 🎯Evaluation
Use the following script to get the PLL score of MLMs:
```
cd code/
python evaluation.py --data [ss, cp] --output ../result/output/ --model [bert-base-cased, roberta-large, albert-large-v2] --sample_rate [sample_rate] --method [aul, cps, sss, gms]
```

For example, if you execute the following script, you will get `result/output/ss_gms_bert-base-cased.json` to record the PLL score.
```
python evaluation.py --data ss --output ../result/output/ --model bert-base-cased --sample_rate 1 --method gms
```
If you set `[sample_rate]` to 0.8, the file name will be `result/output/0.8_ss_gms_bert-base-cased.json`

## 📄Scoring
Use the following script to score the MLM with the PLL score:
```
cd code/
python scoring.py --data [ss, cp] --output ../result/output/ --model [bert-base-cased, roberta-large, albert-large-v2] --sample_rate [sample_rate] --method [aul, sss, cps, kls, jss]
```
For example, if you execute the following script, you will get the `result/scoring/ss_kls_bert-base-cased.txt` record bias score.

```
python scoring.py --data ss --output ../result/output/ --model bert-base-cased --sample_rate 1 --method kls
```
Similarly, if you set `[sample_rate]` to 0.8, the file name will be `result/scoring/0.8_ss_kls_bert-base-cased.json`

If this work has helped you in any way, please cite it by the following:
```
@article{liu2024robust,
    title = {Robust Evaluation Measures for Evaluating Social Biases in Masked Language Models},
    author = {Yang Liu},
    journal = {arXiv preprint arXiv:2401.11601},
    year = {2024},
    doi = {10.48550/arXiv.2401.11601}
}
```
