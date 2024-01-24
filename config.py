TI = {
    'pretrained_model_name_or_path': "/mnt/d/model/stable-diffusion-v1-5/",
    # stable-diffusion-v1-5 / stable-diffusion-2-1
    'emotion': "all",
    'train_data_dir': "/mnt/d/dataset/EmoSet/0103_split_to_folder/",
    'learnable_property': property,
    'max_train_steps': None,
    "num_train_epochs": 1,
    'output_dir': 'TI_emotion',
    'model': 'MLP',
    'learning_rate': 1.0e-03,
    'num_fc_layers': 2,
    'need_LN': False,
    'need_ReLU': True,
    "threshold": 0,
    'need_Dropout': False,
    'attr_rate': 0.01,
    'emo_rate': 0,
    'seed': 633
}

Prior = {
    'prior_model_id': "kakaobrain/karlo-v1-alpha",
    'stable_unclip_model_id': "stabilityai/stable-diffusion-2-1-unclip-small",
    'torch_dtype': 'fp16',
    'variant': "fp16",
}
