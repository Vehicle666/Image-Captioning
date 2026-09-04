from transformers import CLIPTokenizer

from src.config import TOKENIZER_MODEL


class CaptionTokenizer:
    def __init__(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_MODEL, local_files_only=True)

        # CLIP's default pad_token_id == eos_token_id == 49407, which breaks
        # teacher-forcing (EOS targets get ignored by the padding-mask CE loss,
        # so the model never learns to emit EOS). Add a real [PAD] token so that
        # pad != eos. This adds one new vocab entry (49408).
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.sos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def encode(self, text):
        ids = self.tokenizer(text, truncation=True, max_length=77)["input_ids"]
        return ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def decode_tokens(self, ids):
        return [self.tokenizer.decode([i], skip_special_tokens=True) for i in ids]

    def vocab_size(self):
        return len(self.tokenizer)

    def __len__(self):
        return len(self.tokenizer)

    def save(self, path):
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path):
        tok = cls()
        tok.tokenizer = CLIPTokenizer.from_pretrained(path)
        tok.sos_token_id = tok.tokenizer.bos_token_id
        tok.eos_token_id = tok.tokenizer.eos_token_id
        tok.pad_token_id = tok.tokenizer.pad_token_id
        return tok
