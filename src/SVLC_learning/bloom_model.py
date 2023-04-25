import os
import torch
import torch.nn as nn
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BloomConfig
from transformers.models.bloom.modeling_bloom import BloomBlock, build_alibi_tensor


def get_state_dict(model_path, shard_num, prefix=None):
    try:
        d = torch.load(os.path.join(model_path, f"pytorch_model_{shard_num:05d}-of-00072.bin"))
    except:
        print('gg')
    return d if prefix is None else OrderedDict((k.replace(prefix, ''), v) for k, v in d.items())

class Bloom(object):
    def __init__(self) -> None:
        self.model_path = "path" # replace with your local folder path
        self.config = BloomConfig.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.device = 'cuda'
        self.final_lnorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_epsilon, dtype=torch.bfloat16)
        self.final_lnorm.load_state_dict(get_state_dict(self.model_path, shard_num=72, prefix="ln_f."))
        self.final_lnorm.to(self.device)
        self.block = BloomBlock(self.config, layer_number=1).bfloat16()

    def load_embeddings(self):
        state_dict = get_state_dict(self.model_path,shard_num=1, prefix="word_embeddings_layernorm.")
        embeddings = nn.Embedding.from_pretrained(state_dict.pop('word_embeddings.weight'))
        lnorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_epsilon, dtype=torch.bfloat16)
        lnorm.load_state_dict(state_dict)
        return embeddings.to(self.device), lnorm.to(self.device)

    def load_causal_lm_head(self):
        linear = nn.utils.skip_init(
            nn.Linear, self.config.hidden_size, self.config.vocab_size, bias=False, dtype=torch.bfloat16)
        linear.load_state_dict(get_state_dict(self.model_path, shard_num=1, prefix="word_embeddings."), strict=False)
        return linear.bfloat16().to(self.device)

    def load_block(self,block_obj, block_num):
        block_obj.load_state_dict(get_state_dict(self.model_path, shard_num=block_num + 2, prefix=f"h.{block_num}."))
        block_obj.to(self.device)

    def forward(self,input_ids):
        # 1. Create attention mask and position encodings
        attention_mask = torch.ones(len(input_ids)).unsqueeze(0).bfloat16().to(self.device)
        alibi = build_alibi_tensor(input_ids.shape[1], self.config.num_attention_heads,
                                   torch.bfloat16).to(self.device)
        # 2. Load and use word embeddings
        embeddings, lnorm = self.load_embeddings()
        hidden_states = lnorm(embeddings(input_ids))
        del embeddings, lnorm

        # 3. Load and use the BLOOM blocks sequentially
        for block_num in range(70):
            self.load_block(self.block, block_num)
            hidden_states = self.block(hidden_states, attention_mask=attention_mask, alibi=alibi)[0]
            print(".", end='')

        hidden_states = self.final_lnorm(hidden_states)

        # 4. Load and use language model head
        lm_head = self.load_causal_lm_head()
        logits = lm_head(hidden_states)

        # 5. Compute next token
        return torch.argmax(logits[:, -1, :], dim=-1)
