# Bert multi-label text classification by PyTorch

This repo contains a PyTorch implementation of a pretrained BERT model  for multi-label text classification.



## Structure of the code

At the root of the project, you will see:

```text
├── pybert
|  └── callback
|  |  └── lrscheduler.py　　
|  |  └── trainingmonitor.py　
|  |  └── ...
|  └── config
|  |  └── basic_config.py #a configuration file for storing model parameters
|  └── dataset　　　
|  └── io　　　　
|  |  └── dataset.py　　
|  |  └── data_transformer.py　　
|  └── model
|  |  └── nn　
|  |  └── pretrain　
|  └── output #save the ouput of model
|  └── preprocessing #text preprocessing 
|  └── train #used for training a model
|  |  └── trainer.py 
|  |  └── ...
|  └── utils # a set of utility functions
├── convert_tf_checkpoint_to_pytorch.py
├── train_bert_multi_label.py
├── inference.py
```
## Dependencies

- csv
- tqdm
- numpy
- pickle
- scikit-learn
- PyTorch 1.0
- matplotlib
- pandas
- pytorch_pretrained_bert (load bert model)

## Get pytorch bert pretrained model

 Download pretrained bert model (`uncased_L-12_H-768_A-12`)

1. Download the Bert pretrained model from [Google](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and place it into the `/pybert/model/pretrain` directory.

2. `pip3 install pytorch-pretrained-bert` from [github](https://github.com/huggingface/pytorch-pretrained-BERT).

3. Run `python3 convert_tf_checkpoint_to_pytorch.py` to transfer the pretrained model(tensorflow version)  into pytorch form .

4. After the first three steps, check your `model/pretrain/` directory:

   ├── pretrain
   |  └── uncased_L-12_H-768_A-12

   |  |  └── ...

   |  └── pytorch_pretrain

   |  |  └── ...　

   

   ## How to use the code

5. Prepare data. I have left a sample file named `train.tsv` in `pybert/dataset/raw/`.  You can modify the `io.data_transformer.py` to adapt your data, or adjust your format to mine.

      Notes: 

   * No need to split train and test.  No need to shuffle them at first.
   * All sentences are randomly suffled, and split into `train.tsv` and `test.tsv` during the first step of the training. They are stored in `pybert/dataset/processed/`
   * The index doesn't matter. But you can keep it to mark something.
   * One-hot representation.

6. Modify configuration information in `pybert/config/basic_config.py`(the path of data,hyperparameters...).

7. Run `python3 train_bert_multi_label.py` to fine tuning bert model.

8. Run `python3 inference.py` to predict new data, the save path can be edited in `inference.py`.



### result

run:

```python
python3 train_bert_multi_label.py
```

## Tips

- When converting the tensorflow checkpoint into the pytorch, it's expected to choice the "bert_model.ckpt", instead of "bert_model.ckpt.index", as the input file. Otherwise, you will see that the model can learn nothing and give almost same random outputs for any inputs. This means, in fact, you have not loaded the true ckpt for your model
- When using multiple GPUs, the non-tensor calculations, such as accuracy and f1_score, are not supported by DataParallel instance
- The pretrained model has a limit for the sentence of input that its length should is not larger than 512, the max position embedding dim. The data flows into the model as: Raw_data -> WordPieces -> Model. Note that the length of wordPieces is generally larger than that of raw_data, so a safe max length of raw_data is at ~128 - 256 
- Upon testing, we found that fine-tuning all layers could get much better results than those of only fine-tuning the last classfier layer. The latter is actually a feature-based way 
