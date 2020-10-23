# EMNLP2020
full data and images are uploading....


# Datasets
## Snopes
- [x] Testing scenario 1 (SC1): https://drive.google.com/file/d/1S_WWvU1Q1bKElJ04E3MI7z_bLzPIPw5C/view?usp=sharing
- [x] Testing scenario 2 (SC2): https://drive.google.com/file/d/1VDtJk_C-pZtBQXon2jvp4NTyxUnDv-gY/view?usp=sharing
- [x] Augmented data for MAN-A: https://drive.google.com/file/d/1GDONqAZ5lllmF-_XMgk4gVnJNyLP079v/view?usp=sharing
## PolitiFact
- [x] Testing scenario 1 (SC1): https://drive.google.com/file/d/1zeqlv3JeBn-ygn0juTO4SWBucZXIMKZi/view?usp=sharing
- [x] Testing scenario 2 (SC2): https://drive.google.com/file/d/1UDPJdnawYZiicx02shywYGQ3c091Q8xW/view?usp=sharing
- [x] Augmented data for MAN-A: https://drive.google.com/file/d/10e1JhhbfQWYILkovaeopGuhD1VQ_ZPYc/view?usp=sharing 

## Images data
- [x] Extracted features from ResNet50: https://drive.google.com/file/d/17clyyiWyMDMUl6KqrDGGZCi2ZUeNSimh/view?usp=sharing
- [x] Raw images extracted from tweets and fact-checking articles: https://drive.google.com/file/d/11sxoTJx49TBOde_xFY-fgWcG-aHNFhAp/view?usp=sharing


# Usage
## [Step 1] Install required packages
```
pip install requirements.txt
```
## [Step 2] Download and extract images data 
```
pip install gdown
cd EMNLP2020
gdown https://drive.google.com/uc?id=17clyyiWyMDMUl6KqrDGGZCi2ZUeNSimh
unzip images_data.zip
```
## [Step 3.1] Running SC1 (Table 2 in our paper)
### For Snopes
```
gdown https://drive.google.com/uc?id=1S_WWvU1Q1bKElJ04E3MI7z_bLzPIPw5C
python Masters/master_man.py --article_mapped="formatted_data/Snopes/article_mapped.json" --attention_type=3 --batch_size=16 --conv_layers=2 --cuda=1 --elmo_feats="formatted_data/Snopes/elmo_features_avoid_bias" --filters=256 --filters_count_pacrr=16 --fixed_length_left=100 --fixed_length_right=1000 --head_cnn_type="pacrr_plane" --left_images_features="images_data/full_images_otweet_DataC_extracted_features.pth" --log="logs/man" --loss_type="hinge" --lr=0.001 --max_ngram=1 --n_img_in_doc=17 --n_img_in_query=4 --n_s=48 --num_neg=3 --path="formatted_data/Snopes/50_candidates_bm25_extended_reranking_and_text_in_img_avoid_bias" --query_mapped="formatted_data/Snopes/query_mapped.json" --right_images_features="images_data/full_Snopes_extracted_features.pth" --use_elmo=1 --use_visual=1							 
```
### For PolitiFact
```
gdown https://drive.google.com/uc?id=1zeqlv3JeBn-ygn0juTO4SWBucZXIMKZi
python Masters/master_man.py --article_mapped="formatted_data/Politifact/article_mapped.json" --attention_type=3 --batch_size=16 --conv_layers=2 --cuda=1 --elmo_feats="formatted_data/Politifact/elmo_features_avoid_bias" --filters=256 --filters_count_pacrr=16 --fixed_length_left=100 --fixed_length_right=1000 --head_cnn_type="pacrr_plane" --left_images_features="images_data/full_images_otweet_DataC_extracted_features.pth" --log="logs/man" --loss_type="hinge" --lr=0.001 --max_ngram=1 --n_img_in_doc=17 --n_img_in_query=4 --n_s=48 --num_neg=3 --path="formatted_data/Snopes/50_candidates_bm25_extended_reranking_and_text_in_img_avoid_bias" --query_mapped="formatted_data/Snopes/query_mapped.json" --right_images_features="images_data/full_Snopes_extracted_features.pth" --use_elmo=1 --use_visual=1							 
```
## [Step 3.2] Running SC2 (Table 3 in our paper)
### For Snopes dataset
```
gdown https://drive.google.com/uc?id=1VDtJk_C-pZtBQXon2jvp4NTyxUnDv-gY
python Masters/master_man.py --article_mapped="formatted_data/Snopes/article_mapped.json" --attention_type=3 --batch_size=16 --conv_layers=2 --cuda=1 --elmo_feats="formatted_data/Snopes/elmo_features_avoid_bias" --filters=256 --filters_count_pacrr=16 --fixed_length_left=100 --fixed_length_right=1000 --head_cnn_type="pacrr_plane" --left_images_features="images_data/full_images_otweet_DataC_extracted_features.pth" --log="logs/man" --loss_type="hinge" --lr=0.001 --max_ngram=1 --n_img_in_doc=17 --n_img_in_query=4 --n_s=48 --num_neg=3 --path="formatted_data/Snopes/50_candidates_bm25_extended_reranking_and_text_in_img_avoid_bias" --query_mapped="formatted_data/Snopes/query_mapped.json" --right_images_features="images_data/full_Snopes_extracted_features.pth" --use_elmo=1 --use_visual=1							 
```
### For Politifact dataset
```
gdown https://drive.google.com/uc?id=1UDPJdnawYZiicx02shywYGQ3c091Q8xW
python Masters/master_man.py --article_mapped="formatted_data/Politifact/article_mapped.json" --attention_type=3 --batch_size=16 --conv_layers=2 --cuda=1 --elmo_feats="formatted_data/Politifact/elmo_features_avoid_bias" --filters=256 --filters_count_pacrr=16 --fixed_length_left=100 --fixed_length_right=1000 --head_cnn_type="pacrr_plane" --left_images_features="images_data/full_images_otweet_DataC_extracted_features.pth" --log="logs/man" --loss_type="hinge" --lr=0.001 --max_ngram=1 --n_img_in_doc=17 --n_img_in_query=4 --n_s=48 --num_neg=3 --path="formatted_data/Snopes/50_candidates_bm25_extended_reranking_and_text_in_img_avoid_bias" --query_mapped="formatted_data/Snopes/query_mapped.json" --right_images_features="images_data/full_Snopes_extracted_features.pth" --use_elmo=1 --use_visual=1							 
```

# Citation
Please cite our work as follows:

```
@inproceedings{vo2020facts,
	title={Where Are the Facts? Searching for Fact-checked Information 
	       to Alleviate the Spread of Fake News},
	author={Vo, Nguyen and Lee, Kyumin},
	booktitle={Proceedings of the 2020 Conference on Empirical Methods 
	           in Natural Language Processing (EMNLP 2020)},
	year={2020}
}
```
