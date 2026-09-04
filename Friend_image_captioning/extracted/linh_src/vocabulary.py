from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from linh_src.config import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, SPECIAL_TOKENS, UNK_TOKEN


class CaptionTokenizer:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    @classmethod
    def train(cls, captions, vocab_size=5000, min_frequency=2, save_path=None):
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single=f"<SOS>:0 $A:0 <EOS>:0",
            special_tokens=[("<SOS>", SOS_TOKEN), ("<EOS>", EOS_TOKEN)],
        )

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            continuing_subword_prefix="##",
        )
        tokenizer.train_from_iterator(captions, trainer=trainer)

        if save_path:
            tokenizer.save(save_path)

        return cls(tokenizer)

    @classmethod
    def load(cls, path):
        tokenizer = Tokenizer.from_file(path)
        return cls(tokenizer)

    def encode(self, text):
        output = self.tokenizer.encode(text)
        return output.ids

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def add_tokens(self, new_tokens):
        return self.tokenizer.add_tokens(new_tokens)

    @property
    def pad_token_id(self):
        return PAD_TOKEN

    @property
    def sos_token_id(self):
        return SOS_TOKEN

    @property
    def eos_token_id(self):
        return EOS_TOKEN

    @property
    def unk_token_id(self):
        return UNK_TOKEN

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def __repr__(self):
        return f"CaptionTokenizer(vocab_size={len(self)})"
