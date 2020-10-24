# EMNLP2020
This is the repository to reproduce results in the paper
"Where Are the Facts? Searching for Fact-checked Information to Alleviate the Spread of Fake News", EMNLP 2020.  
# Multimodal Attention Network
![alt text](https://github.com/nguyenvo09/EMNLP2020/blob/master/examples/man.png)

# Datasets
## Snopes
- Testing scenario 1 (SC1): https://drive.google.com/file/d/1S_WWvU1Q1bKElJ04E3MI7z_bLzPIPw5C/view?usp=sharing
- Testing scenario 2 (SC2): https://drive.google.com/file/d/1VDtJk_C-pZtBQXon2jvp4NTyxUnDv-gY/view?usp=sharing
- Augmented data for MAN-A: https://drive.google.com/file/d/1GDONqAZ5lllmF-_XMgk4gVnJNyLP079v/view?usp=sharing
## PolitiFact
- Testing scenario 1 (SC1): https://drive.google.com/file/d/1zeqlv3JeBn-ygn0juTO4SWBucZXIMKZi/view?usp=sharing
- Testing scenario 2 (SC2): https://drive.google.com/file/d/1UDPJdnawYZiicx02shywYGQ3c091Q8xW/view?usp=sharing
- Augmented data for MAN-A: https://drive.google.com/file/d/10e1JhhbfQWYILkovaeopGuhD1VQ_ZPYc/view?usp=sharing 

## Images data
- Extracted features from ResNet50: https://drive.google.com/file/d/17clyyiWyMDMUl6KqrDGGZCi2ZUeNSimh/view?usp=sharing
- Raw images extracted from tweets and fact-checking articles: https://drive.google.com/file/d/11sxoTJx49TBOde_xFY-fgWcG-aHNFhAp/view?usp=sharing

## Interesting misinformation and corresponding fact-checking articles in our datasets:
### Example 1: President Trump 
![alt text](https://github.com/nguyenvo09/EMNLP2020/blob/master/examples/trump.png)
### Example 2: Vice president Biden
![alt text](https://github.com/nguyenvo09/EMNLP2020/blob/master/examples/biden.png)
## Structure of dataset folders
After downloading and extracting data, the expected structure of `formatted_data` is as follows:
```
EMNLP2020/
├── formatted_data
│   ├── Politifact
│   │   ├── 50_candidates_bm25_extended_reranking
│   │   │   ├── Politifact.dev.tsv
│   │   │   ├── Politifact.test.tsv
│   │   │   ├── Politifact.test2_hard.tsv
│   │   │   └── Politifact.train.tsv
│   │   ├── 50_candidates_bm25_extended_reranking_and_text_in_img
│   │   │   ├── Politifact.dev.tsv
│   │   │   ├── Politifact.test.tsv
│   │   │   ├── Politifact.test2_hard.tsv
│   │   │   └── Politifact.train.tsv
│   │   ├── 50_candidates_bm25_extended_reranking_and_text_in_img_avoid_bias
│   │   │   ├── Politifact.dev.tsv
│   │   │   ├── Politifact.test.tsv
│   │   │   ├── Politifact.test2_hard.tsv
│   │   │   └── Politifact.train.tsv
│   │   ├── article_mapped.json
│   │   ├── articles_content.json
│   │   ├── elmo_features_avoid_bias
│   │   │   ├── articles_feats.pth
│   │   │   └── queries_feats.pth
│   │   ├── elmo_features_only_text_in_tweets
│   │   │   ├── articles_feats.pth
│   │   │   └── queries_feats.pth
│   │   ├── elmo_features_use_text_in_img
│   │   │   ├── articles_feats.pth
│   │   │   └── queries_feats.pth
│   │   ├── queries_content.json
│   │   ├── query.negatives
│   │   ├── query_article_interaction.csv
│   │   └── query_mapped.json
│   └── Snopes
│       ├── 50_candidates_bm25_extended_reranking
│       │   ├── Snopes.dev.tsv
│       │   ├── Snopes.test.tsv
│       │   ├── Snopes.test2_hard.tsv
│       │   └── Snopes.train.tsv
│       ├── 50_candidates_bm25_extended_reranking_and_text_in_img
│       │   ├── Snopes.dev.tsv
│       │   ├── Snopes.test.tsv
│       │   ├── Snopes.test2_hard.tsv
│       │   └── Snopes.train.tsv
│       ├── 50_candidates_bm25_extended_reranking_and_text_in_img_avoid_bias
│       │   ├── Snopes.dev.tsv
│       │   ├── Snopes.test.tsv
│       │   ├── Snopes.test2_hard.tsv
│       │   └── Snopes.train.tsv
│       ├── article_mapped.json
│       ├── articles_content.json
│       ├── elmo_features_avoid_bias
│       │   ├── articles_feats.pth
│       │   └── queries_feats.pth
│       ├── elmo_features_only_text_in_tweets
│       │   ├── articles_feats.pth
│       │   └── queries_feats.pth
│       ├── elmo_features_use_text_in_img
│       │   ├── articles_feats.pth
│       │   └── queries_feats.pth
│       ├── queries_content.json
│       ├── query.negatives
│       ├── query_article_interaction.csv
│       └── query_mapped.json
├── images_data
│   ├── full_Snopes_extracted_features.pth
│   ├── full_images_otweet_DataC_extracted_features.pth
│   ├── resnet50_Politifact_documents_extracted_features.pth
│   └── resnet50_Polititact_queries_extracted_features.pth
```
# Usage
## 1. Install required packages
We use Pytorch 0.4.1 and python 3.5. 
```
pip install requirements.txt
```
## 2. Download and extract images data 
```
pip install gdown
cd EMNLP2020
gdown https://drive.google.com/uc?id=17clyyiWyMDMUl6KqrDGGZCi2ZUeNSimh
unzip images_data.zip
rm images_data.zip
```
If you want to see raw images, you can download it as follows:
```
gdown https://drive.google.com/u/0/uc?id=11sxoTJx49TBOde_xFY-fgWcG-aHNFhAp
unzip raw_images.zip
```
## 3.1 Running SC1 (Table 2 in our paper)
### For Snopes
```
gdown https://drive.google.com/uc?id=1S_WWvU1Q1bKElJ04E3MI7z_bLzPIPw5C
unzip SC1_snopes.zip -d formatted_data/Snopes
mkdir logs
python Masters/master_man.py --attention_type=4 \
                             --conv_layers=2 \
                             --cuda=1 \
                             --use_elmo=1 --use_visual=1 \
                             --filters=256 \
                             --filters_count_pacrr=16 \
                             --fixed_length_left=50 \
                             --fixed_length_right=1000 \
                             --log="logs/man" \
                             --loss_type="hinge" \
                             --max_ngram=1 \
                             --n_s=48 \
                             --path="formatted_data/Snopes/50_candidates_bm25_extended_reranking" \
                             --query_mapped="formatted_data/Snopes/query_mapped.json" \
                             --article_mapped="formatted_data/Snopes/article_mapped.json" \
                             --left_images_features="images_data/full_images_otweet_DataC_extracted_features.pth" \
                             --right_images_features="images_data/full_Snopes_extracted_features.pth" \
                             --elmo_feats="formatted_data/Snopes/elmo_features_only_text_in_tweets"
```
### For PolitiFact
```
gdown https://drive.google.com/uc?id=1zeqlv3JeBn-ygn0juTO4SWBucZXIMKZi
unzip SC1_politifact.zip -d formatted_data/Politifact
python Masters/master_man.py --attention_type=4 \
                             --conv_layers=2 \
                             --cuda=1 \
                             --use_elmo=1 --use_visual=1 \
                             --filters=256 \
                             --filters_count_pacrr=16 \
                             --fixed_length_left=50 \
                             --fixed_length_right=1000 \
                             --log="logs/man" \
                             --loss_type="hinge" \
                             --max_ngram=1 \
                             --n_s=48 \
                             --path="formatted_data/Politifact/50_candidates_bm25_extended_reranking" \
                             --query_mapped="formatted_data/Politifact/query_mapped.json" \
                             --article_mapped="formatted_data/Politifact/article_mapped.json" \
                             --left_images_features="images_data/resnet50_Polititact_queries_extracted_features.pth" \
                             --right_images_features="images_data/resnet50_Politifact_documents_extracted_features.pth" \
                             --elmo_feats="formatted_data/Politifact/elmo_features_only_text_in_tweets"
```
## 3.2 Running SC2 (MAN in Table 3 in our paper)
### For Snopes dataset
```
gdown https://drive.google.com/uc?id=1VDtJk_C-pZtBQXon2jvp4NTyxUnDv-gY
unzip SC2_snopes.zip -d formatted_data/Snopes
python Masters/master_man.py --attention_type=2 \
                             --conv_layers=2 \
                             --cuda=1 \
                             --use_elmo=1 --use_visual=1 \
                             --filters=256 \
                             --filters_count_pacrr=16 \
                             --fixed_length_left=100 \
                             --fixed_length_right=1000 \
                             --log="logs/man" \
                             --loss_type="hinge" \
                             --max_ngram=1 \
                             --n_s=32 \
                             --path="formatted_data/Snopes/50_candidates_bm25_extended_reranking_and_text_in_img" \
                             --query_mapped="formatted_data/Snopes/query_mapped.json" \
                             --article_mapped="formatted_data/Snopes/article_mapped.json" \
                             --left_images_features="images_data/full_images_otweet_DataC_extracted_features.pth" \
                             --right_images_features="images_data/full_Snopes_extracted_features.pth" \
                             --elmo_feats="formatted_data/Snopes/elmo_features_use_text_in_img"
```
### For Politifact dataset
```
gdown https://drive.google.com/uc?id=1UDPJdnawYZiicx02shywYGQ3c091Q8xW
unzip SC2_politifact.zip -d formatted_data/Politifact
python Masters/master_man.py --attention_type=2 \
                             --conv_layers=3 \
                             --cuda=1 \
                             --use_elmo=1 --use_visual=1 \
                             --filters=256 \
                             --filters_count_pacrr=16 \
                             --fixed_length_left=100 \
                             --fixed_length_right=1000 \
                             --log="logs/man" \
                             --loss_type="hinge" \
                             --max_ngram=1 \
                             --n_s=32 \
                             --path="formatted_data/Politifact/50_candidates_bm25_extended_reranking_and_text_in_img" \
                             --query_mapped="formatted_data/Politifact/query_mapped.json" \
                             --article_mapped="formatted_data/Politifact/article_mapped.json" \
                             --left_images_features="images_data/resnet50_Polititact_queries_extracted_features.pth" \
                             --right_images_features="images_data/resnet50_Politifact_documents_extracted_features.pth" \
                             --elmo_feats="formatted_data/Politifact/elmo_features_use_text_in_img"
```
## 3.3 Running SC2 with augmented data (MAN-A in Table 3 in our paper)
This test is memory-intensive so we recommend to run this test on a server with 64Gb RAM. 
### For Snopes dataset
```
gdown https://drive.google.com/u/0/uc?id=1GDONqAZ5lllmF-_XMgk4gVnJNyLP079v
unzip augment_snopes.zip -d formatted_data/Snopes
python Masters/master_man.py --attention_type=2 \
                             --conv_layers=2 \
                             --cuda=1 \
                             --use_elmo=1 --use_visual=1 \
                             --filters=256 \
                             --filters_count_pacrr=16 \
                             --fixed_length_left=100 \
                             --fixed_length_right=1000 \
                             --log="logs/man" \
                             --loss_type="hinge" \
                             --max_ngram=2 \
                             --n_s=32 \
                             --path="formatted_data/Snopes/50_candidates_bm25_extended_reranking_and_text_in_img_avoid_bias" \
                             --query_mapped="formatted_data/Snopes/query_mapped.json" \
                             --article_mapped="formatted_data/Snopes/article_mapped.json" \
                             --left_images_features="images_data/full_images_otweet_DataC_extracted_features.pth" \
                             --right_images_features="images_data/full_Snopes_extracted_features.pth" \
                             --elmo_feats="formatted_data/Snopes/elmo_features_avoid_bias"
```
### For PolitiFact dataset
```
gdown https://drive.google.com/u/0/uc?id=10e1JhhbfQWYILkovaeopGuhD1VQ_ZPYc
unzip augment_politifact.zip -d formatted_data/Politifact
python Masters/master_man.py --attention_type=4 \
                             --conv_layers=2 \
                             --cuda=1 \
                             --use_elmo=1 --use_visual=1 \
                             --filters=256 \
                             --filters_count_pacrr=16 \
                             --fixed_length_left=100 \
                             --fixed_length_right=1000 \
                             --log="logs/man" \
                             --loss_type="hinge" \
                             --max_ngram=3 \
                             --n_s=48 \
                             --path="formatted_data/Politifact/50_candidates_bm25_extended_reranking_and_text_in_img_avoid_bias" \
                             --query_mapped="formatted_data/Politifact/query_mapped.json" \
                             --article_mapped="formatted_data/Politifact/article_mapped.json" \
                             --left_images_features="images_data/resnet50_Polititact_queries_extracted_features.pth" \
                             --right_images_features="images_data/resnet50_Politifact_documents_extracted_features.pth" \
                             --elmo_feats="formatted_data/Politifact/elmo_features_avoid_bias"
```
# Citation
Please cite our work as follows:

```
@inproceedings{vo2020facts,
	title={Where Are the Facts? Searching for Fact-checked Information to Alleviate the Spread of Fake News},
	author={Vo, Nguyen and Lee, Kyumin},
	booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020)},
	year={2020}
}
```
