import torch

def save_progress(text_encoder, emotion_list, save_path):
    start = text_encoder.vocab_size -8
    for emotion in emotion_list:
        save_name = f"{save_path}/{emotion}.pt"
        learned_embeds = (
            text_encoder
            .get_input_embeddings()
            .weight[start : start + 1]
        )
        learned_embeds_dict = {f"<{emotion}>": learned_embeds.detach().cpu()}
        torch.save(learned_embeds_dict, save_name)
        start += 1

def save_embedding(embedding_list, emotion_list, save_path):
    for emotion in emotion_list:
        embed_list = embedding_list[emotion]
        if len(embed_list) != 0:
            for i, embed in enumerate(embed_list):
                save_name = f"{save_path}/{emotion}_{i}.pt"
                learned_embeds_dict = {f"<{emotion}_{i}>": embed.cpu()}
                torch.save(learned_embeds_dict, save_name)