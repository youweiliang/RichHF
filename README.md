# Rich Human Feedback for Text-to-Image Generation

This repository holds the training and inference code to replicate the results in the CVPR 2024 paper [Rich Human Feedback for Text-to-Image Generation](https://arxiv.org/pdf/2312.10240). The RAHF (Rich Automatic Human Feedback) model here is built as close to the RAHF model in the paper as possible. Most of the testing metrics are reproduced. However, since the ViT and T5X models used in the paper are not exactly the same as the publicly available ViT and T5 models in PyTorch used in this repository, there could be some minor differences between the RAHF model here and the original model developed at Google.

## Environment
The code is tested with Python 3.9.18, torch 2.0.0, torchvision 0.15.1, transformers 4.32.0, datasets 2.16.1, scipy 1.11.4, tensorboard 2.18.0, scikit-learn 1.3.2, and tensorflow 2.18.0. Note: tensorflow is only needed to load the RichHF-18K dataset.

## Data
*If you just want to use the trained RAHF model, you don't need to download this dataset. Check [Trained Models and Results](#trained-models-and-results) below.*

Clone the RichHF-18K dataset from [https://github.com/google-research-datasets/richhf-18k](https://github.com/google-research-datasets/richhf-18k) into this repository.  
Create a folder to download images: `mkdir data`.  

Run `python get_dataset.py` to download the Pick-a-Pic dataset `yuvalkirstain/pickapic_v1` (~190GB) and extract the images needed for RichHF-18K. The extracted images and heatmaps are saved under `richhf-18k/train`, `richhf-18k/dev`, and `richhf-18k/test` for browsing. They are also saved as a HuggingFace dataset under `/data/rich_human_feedback_dataset`.

## Training
After downloading the data, run `python train.py --multi_heads` to train the RAHF model on the RichHF-18K dataset. For other parameters, run `python train.py -h`. Model checkpoints and logs are saved under the `exp` directory by default. To start training with multiple GPUs, run `torchrun --nproc_per_node [number_of_gpus] --master_port [port_number] train.py --multi_heads`.

**Note:** Scaling up the ViT model improves RAHF performance, while scaling up T5 from T5-base to T5-large does not. Thus, we use `google/vit-large-patch16-384` for ViT and `t5-base` for T5.

## Inference
After training, run `python inference.py --log_dir [your_log_path] --infer` to infer and visualize heatmaps on the test set. Run `python inference.py -h` for help.

## Trained Models and Results

A **multi-head** RAHF model checkpoint is available on [Google Drive](https://drive.google.com/file/d/1-jKfmpyGtJ0UAgEQ23zylRsmQ82qigzB/view?usp=sharing). Load the weights into a RAHF model with this configuration: `"vit_model": "google/vit-large-patch16-384", "t5_model": "t5-base"`.

Testing metrics are comparable to the [RichHF paper](https://arxiv.org/pdf/2312.10240), with slight variations:

#### Heatmap Prediction:
| Model          | All Data MSE ↓ | GT=0 MSE ↓ | GT>0 CC ↑ | GT>0 KLD ↓ | GT>0 SIM ↑ | GT>0 NSS ↑ | GT>0 AUC-Judd ↑ |
|----------------|----------------|------------|-----------|------------|------------|------------|-----------------|
| Implausibility | 0.00724        | 0.00078    | 0.525     | 1.649      | 0.339      | 2.029      | 0.905           |
| Misalignment   | 0.00324        | 0.00035    | 0.232     | 2.848      | 0.104      | 1.256      | 0.797           |

#### Score Prediction:
| Metric   | Plausibility | Aesthetics | Text-Image Alignment | Overall |
|----------|--------------|------------|----------------------|---------|
| PLCC ↑   | 0.739        | 0.629      | 0.565                | 0.642   |
| SRCC ↑   | 0.726        | 0.621      | 0.564                | 0.634   |

#### Misaligned Text Prediction:
| Precision | Recall | F1 Score |
|-----------|--------|----------|
| 56.5      | 45.2   | 47.5     |

The augmented-prompt RAHF variant trained here underperforms the multi-head version, contrary to findings in the paper. We attribute this to differences between our implementation and Google’s original setup.

## Acknowledgement
To cite our paper:
```
@inproceedings{richhf,
  title={Rich Human Feedback for Text-to-Image Generation},
  author={Youwei Liang and Junfeng He and Gang Li and Peizhao Li and Arseniy Klimovskiy and Nicholas Carolan and Jiao Sun and Jordi Pont-Tuset and Sarah Young and Feng Yang and Junjie Ke and Krishnamurthy Dj Dvijotham and Katie Collins and Yiwen Luo and Yang Li and Kai J Kohlhoff and Deepak Ramachandran and Vidhya Navalpakkam},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024},
}
```

Parts of the data loading code are from this [repo](https://github.com/google-research/google-research/tree/master/richhf_18k) and this [repo](https://github.com/RAraghavarora/RichHF_T2I).