import datetime
from utils import *
import torch
from diffusers import UnCLIPScheduler, DDPMScheduler
from pipeline_Unclip import StableUnCLIPPipeline_prior_v2
from diffusers.models import PriorTransformer
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from model import *
from args_config import *
import os
from diffusers.optimization import get_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import pickle
from tqdm import tqdm

@torch.no_grad()
def positive(pipe,batch_size):
    postive_prompt = f"Professional high-quality art of photo. photorealistic, 4k, HQ"
    image_embed, prompt_embed = pipe(prompt=postive_prompt, noise_level=0, num_images_per_prompt=batch_size,
                                     prior_num_inference_steps=10)
    return image_embed, prompt_embed
def main(num_picture):
    args = parse_args()
    now = datetime.datetime.now()
    time_str = now.strftime('%Y-%m-%d_%H-%M')
    output_dir = os.path.join("record",time_str)
    writer = SummaryWriter(f'{output_dir}/logs')
    emotion_list = ["amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"]
    # emotion_list = ["amusement", "awe", "contentment"]
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    prior_model_id = "/mnt/d/model/karlo-v1-alpha"
    data_type = torch.float32
    prior = PriorTransformer.from_pretrained(prior_model_id, subfolder="prior", torch_dtype=data_type)

    prior_text_model_id = "/mnt/d/model/clip-vit-large-patch14"
    prior_tokenizer = CLIPTokenizer.from_pretrained(prior_text_model_id)
    prior_text_model = CLIPTextModelWithProjection.from_pretrained(prior_text_model_id, torch_dtype=data_type)
    prior_scheduler = UnCLIPScheduler.from_pretrained(prior_model_id, subfolder="prior_scheduler")
    prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)

    stable_unclip_model_id = "/mnt/d/model/stable-diffusion-2-1-unclip-small/"

    pipe = StableUnCLIPPipeline_prior_v2.from_pretrained(
        stable_unclip_model_id,
        torch_dtype=data_type,
        variant="fp32",
        prior_tokenizer=prior_tokenizer,
        prior_text_encoder=prior_text_model,
        prior=prior,
        prior_scheduler=prior_scheduler,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    # load new_embed
    for emotion in emotion_list:
        pipe.load_textual_inversion(f"textual_inversion_few/neural/tree.bin",
                                    text_encoder=prior_text_model, tokenizer=prior_tokenizer, token=f"<{emotion}>")
        # pipe.load_textual_inversion(f"textual_inversion_few_1024/{emotion}/learned_embeds.safetensors",
        #                             token=f"<{emotion}>")

    # freeze all parameters except for the token embeddings in text encoder
    # pipe.unet.requires_grad_(False)
    #
    # pipe.text_encoder.text_model.encoder.requires_grad_(False)
    # pipe.text_encoder.text_model.final_layer_norm.requires_grad_(False)
    # pipe.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    pipe.prior_text_encoder.text_model.encoder.requires_grad_(False)
    pipe.prior_text_encoder.text_model.final_layer_norm.requires_grad_(False)
    pipe.prior_text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    # Initialize the optimizer
    # updata_parameter = list(pipe.text_encoder.get_input_embeddings().parameters()) + \
    #                    list(pipe.prior_text_encoder.get_input_embeddings().parameters())

    optimizer = torch.optim.AdamW(
        pipe.prior_text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Initialize the lr_scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles * args.gradient_accumulation_steps,
    )

    # Initialize the emotion_classifier
    emotion_classifier = emo_classifier_768().to(pipe.device)
    state_dict = torch.load('Clip_emotion_classifier/weights/2023-11-12-best.pth', map_location=pipe.device)
    # for key in state_dict.keys():
    #     state_dict[key] = state_dict[key].half()
    emotion_classifier.load_state_dict(state_dict)
    # emotion_classifier.half()
    emotion_classifier.eval()

    # define loss function
    loss_fn = nn.CrossEntropyLoss()

    # keep original embeddings as reference
    prior_orig_embeds_params = pipe.prior_text_encoder.get_input_embeddings().weight.data.clone()

    # orig_embeds_params = pipe.text_encoder.get_input_embeddings().weight.data.clone()

    def print_grad(grad):
        print(grad)

    token_list = {emo: [] for emo in emotion_list}
    img_list = {emo: [] for emo in emotion_list}
    batch_size = num_picture
    train_steps = args.epochs * len(emotion_list)
    progress_bar = tqdm(range(0, train_steps), ncols=200)
    progress_bar.set_description("Steps")
    positive_image_embed, positive_prompt_embed = positive(pipe, 1)

    for epoch in range(args.epochs):
        pipe.prior_text_encoder.train()
        totol_loss = []
        for index, emotion in enumerate(emotion_list):
            label = torch.tensor([index] * batch_size)
            wave_prompt = f"Professional high-quality art of a <{emotion}>. photorealistic, 4k, HQ"
            # prompt = A vivid picture, displaying detailed and realistic details
            image_embed,_ = pipe(prompt=wave_prompt, noise_level=0, num_images_per_prompt=batch_size,
                               prior_num_inference_steps=10)
            pred = emotion_classifier(image_embed)
            loss = loss_fn(pred, label.to(pred.device))

            mean_imgae_embed = torch.mean(image_embed,dim=0,keepdim=True)
            mean_imgae_embed_norm = mean_imgae_embed/mean_imgae_embed.norm(p=2, dim=-1, keepdim=True)
            token_loss = torch.tensor([0.0], requires_grad=True).to(pred.device)
            if len(token_list[emotion]) != 0:
                token_loss = torch.max(F.cosine_similarity(mean_imgae_embed_norm, torch.cat(img_list[emotion])))
            positive_loss = torch.tensor([0.0], requires_grad=True).to(pred.device)
            positive_loss += (1 - F.cosine_similarity(mean_imgae_embed_norm, positive_prompt_embed[-1]))
            sum_loss = loss + 0.5*token_loss + positive_loss

            if (epoch+1) % 100 == 0:
                token_id = prior_tokenizer.added_tokens_encoder[f"<{emotion}>"]
                cor_embed = pipe.prior_text_encoder.get_input_embeddings().weight[token_id].detach().clone().view(1,-1)
                token_list[emotion].append(cor_embed)
                img_list[emotion].append(mean_imgae_embed_norm.detach().clone().view(1,-1))
            # pipe.prior_text_encoder.get_input_embeddings().weight.register_hook(print_grad)
            sum_loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            index_no_updates = torch.ones((len(pipe.prior_tokenizer),), dtype=torch.bool)
            index_no_updates[(len(pipe.prior_tokenizer) - 8): (len(pipe.prior_tokenizer))] = False

            with torch.no_grad():
                pipe.prior_text_encoder.get_input_embeddings().weight[
                    index_no_updates
                ] = prior_orig_embeds_params[index_no_updates]
                totol_loss.append(loss.detach().item())
                writer.add_scalar(f'{emotion}_loss', loss.detach().item(), global_step=epoch)
                writer.add_scalar(f'{emotion}_token_loss', token_loss.detach().item(), global_step=epoch)

                logs = {"sum_loss": sum_loss.detach().item(), "loss":loss.detach().item(),
                        "token_loss": token_loss.detach().item(), "epoch":epoch}
                progress_bar.set_postfix(**logs)
                progress_bar.update(1)
        totol_loss = sum(totol_loss)/len(totol_loss)
        writer.add_scalar(f'total_loss', totol_loss, global_step=epoch)
        if (epoch+1) % args.save_per_epoch == 0:
            save_path = os.path.join(output_dir, "dictionary")
            os.makedirs(save_path, exist_ok=True)
            save_embedding(token_list,emotion_list,save_path)
                # pipe.text_encoder.get_input_embeddings().weight[
                #     index_no_updates
                # ] = orig_embeds_params[index_no_updates]
    # save_path = os.path.join(output_dir, "final")
    # os.makedirs(save_path, exist_ok=True)
    # save_embedding(token_list, emotion_list, save_path)
    writer.close()
if __name__ == "__main__":
    num_picture = 2
    main(num_picture)
