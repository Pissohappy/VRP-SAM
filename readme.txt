CrossAttention定义：model/base/transformer_decoder.py里，support → query 初始化和query → target 交互均使用

Gated Fusion需求：在 transformer_decoder 之前，或者在 protos 输出之后，插入一个 文本特征 Qt，再和当前视觉特征 / 原型做门控融合。

修改范围：
1. VRP_encoder.py:
__init__里新增文本编码和 fusion 模块
forward()修改
transformer_decoder后面接Gated Fusion
封装fusion函数
2. train.py:
模型调用传入class_id

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 train.py \
    --benchmark pascal \
    --datapath /home/user/zhangdantong/Datasets_HSN \
    --bsz 1 \
    --text_prompt_template "a segmentation mask of a {class_name}"
