import torch

from src.config import BEAM_SIZE, device


@torch.no_grad()
def generate_caption(model, image, tokenizer, max_length=50):
    features = model.encoder(image)
    caption = [tokenizer.sos_token_id]
    for _ in range(max_length):
        tgt = torch.tensor([caption]).to(device)
        logits = model.decoder(features, tgt)
        pred = logits[0, -1, :].argmax().item()
        if pred == tokenizer.eos_token_id:
            break
        caption.append(pred)
    return caption[1:]


@torch.no_grad()
def generate_caption_beam(model, image, tokenizer, beam_size=BEAM_SIZE, max_length=50):
    features = model.encoder(image)
    start_token = tokenizer.sos_token_id
    end_token = tokenizer.eos_token_id

    sequences = [[start_token]]
    scores = [0.0]

    for _ in range(max_length):
        all_candidates = []
        for seq, score in zip(sequences, scores):
            if seq[-1] == end_token:
                all_candidates.append((seq, score))
                continue

            tgt = torch.tensor([seq]).to(device)
            logits = model.decoder(features, tgt)
            probs = torch.softmax(logits[0, -1, :], dim=0)
            topk_probs, topk_indices = torch.topk(probs, beam_size)

            for i in range(beam_size):
                all_candidates.append(
                    (seq + [topk_indices[i].item()], score + torch.log(topk_probs[i]).item())
                )

        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = [seq for seq, _ in ordered[:beam_size]]
        scores = [s for _, s in ordered[:beam_size]]

        if all(seq[-1] == end_token for seq in sequences):
            break

    best = sequences[0][1:]
    if best and best[-1] == end_token:
        best = best[:-1]
    return best
