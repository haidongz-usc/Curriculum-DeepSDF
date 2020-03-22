# Curriculum-DeepSDF

This repository is an implementation for Curriculum DeepSDF. Full paper is available [here](https://arxiv.org/abs/2003.08593).

### Preparation

Please follow original setting of [DeepSDF](https://github.com/facebookresearch/DeepSDF) to prepare the data.

### Usage

After preparing the data following DeepSDF, you can train and test the model as follow.

```
# Train a Curriculum DeepSDF model 
CUDA_VISIBLE_DEVICES=${gpu_id} python train_deep_sdf.py -e examples/${cat_name}

# Reconstruct the meshes with models
CUDA_VISIBLE_DEVICES=${gpu_id} python reconstruct.py -e examples/${cat_name} -c 2000 --split examples/splits/sv2_${cat_name}_test.json -d ${data_dir} --skip

# Evaluate the reconstructions
python evaluate.py -e examples/${cat_name}  -c 2000 -d ${data_dir} -s examples/splits/sv2_${cat_name}_test.json
#Replace evaluate.py with eval2.py if you want to do multiprocessing evaluation.
``` 

We uploaded our pretrained model for the category 'lamps' in its folder, as well as our reconstruction results for [lamps](https://drive.google.com/file/d/1JhIhQXBwaaPCaQptXL7PcyBLY-VD_w8y/view?usp=sharing) using this pretrained model. If you successfully have the data preprocessed following DeepSDF, the average of the Chamfer distances for it should be around 0.000473 (small variance is due to the randomness of the points sampled from the mesh).
