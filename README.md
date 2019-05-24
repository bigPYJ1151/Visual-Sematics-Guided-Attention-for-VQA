# Visual-Sematics-Guided-Attention-for-VQA
## Graduation Design for Undergraduates
This model based on [VQA_demo_Pytorch][0], and proposed Visual Semantics Guided Attention, a buttom-up attention based on semantic segmentation.

Visual Semantics Guided Attention uses [BiSeNet][1] to extract image semantic information.

This model is trained on [VQA][2] *Openend-RealImages* train set, and get 64.88% accuracy on standard-test dataset.

## Detials
`config.yaml` contains some setting parameters.

`image_feature.py` implement a extractor to extract image features. Use the pretrained Resnet-152 provided by torchvision.

Use `vocab_extract.py` to get question vocabulary and answer set.

Use `BiSeNet.py` to train BiSeNet on COCO dataset.

Use `train.py` and `eval.py` to train and test model.

[0]:https://github.com/bigPYJ1151/VQA_demo_Pytorch
[1]:https://arxiv.org/abs/1808.00897
[2]:https://visualqa.org/vqa_v1_download.html
