"""BioCLIP (LoRA) encoder -> projector -> Gemma 3 (LoRA) decoder."""
import math

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

from utils import SpeciesDone


class GemmaTraitGen(nn.Module):
    def __init__(self, lm_id, clip_id, device="cuda", n_img_tokens=49):
        super().__init__()
        self.device = device
        self.n_img_tokens = n_img_tokens

        # tokenizer: right padding (Gemma defaults to left), explicit pad token
        self.tokenizer = AutoTokenizer.from_pretrained(lm_id)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- vision: BioCLIP in fp16, LoRA on MLPs of last 6 blocks
        clip, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(clip_id)
        visual = clip.visual
        del clip
        visual.proj = None            # native 768-d tokens
        visual.output_tokens = True
        visual = visual.to(device, dtype=torch.float16)
        for p in visual.parameters():
            p.requires_grad = False
        enc_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.2, bias="none",
            target_modules=r".*resblocks\.(6|7|8|9|10|11)\.mlp\.(c_fc|c_proj)",
        )
        self.visual = get_peft_model(visual, enc_cfg)

        # ---- decoder: FP32 weights, eager attention (Gemma 3 overflows in fp16)
        lm = AutoModelForCausalLM.from_pretrained(
            lm_id, dtype=torch.float32, attn_implementation="eager"
        ).to(device)
        hidden = lm.config.hidden_size

        # RMS of real (scaled) text embeddings -> used to scale the image tokens
        embed = lm.get_input_embeddings()
        with torch.no_grad():
            es = getattr(embed, "embed_scale", None)
            es = float(es) if es is not None else math.sqrt(hidden)
            text_rms = float(embed.weight[:20000].float().pow(2).mean().sqrt() * es)

        dec_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.2, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.lm = get_peft_model(lm, dec_cfg)

        # trainable params must be fp32 for GradScaler
        for m in (self.visual, self.lm):
            for n, p in m.named_parameters():
                if "lora_" in n:
                    p.data = p.data.float()
                    p.requires_grad = True

        # ---- projector: LN -> MLP -> LN(no affine) * learnable scale (init = text embedding RMS)
        self.vis_norm = nn.LayerNorm(768)
        self.projector = nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.out_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.out_scale = nn.Parameter(torch.tensor(text_rms))
        self.vis_norm.to(device)
        self.projector.to(device)
        self.out_norm.to(device)
        self.out_scale.data = self.out_scale.data.to(device)

    def projector_params(self):
        return list(self.vis_norm.parameters()) + list(self.projector.parameters()) + [self.out_scale]

    def encode_images(self, pixels):
        with torch.autocast("cuda", dtype=torch.float16):
            _, tokens = self.visual(pixels.to(self.device, dtype=torch.float16))   # (B,196,768)
        tokens = tokens.float()
        B, N, D = tokens.shape
        s = int(math.sqrt(N))
        grid = tokens.transpose(1, 2).reshape(B, D, s, s)
        side = int(math.sqrt(self.n_img_tokens))
        pooled = F.adaptive_avg_pool2d(grid, side).flatten(2).transpose(1, 2)       # (B,49,768)
        with torch.autocast("cuda", enabled=False):
            x = self.projector(self.vis_norm(pooled))
            return self.out_norm(x) * self.out_scale                                 # fp32

    @torch.no_grad()
    def generate(self, pixels, max_new_tokens=120, stop_after_species=True):
        """Greedy-decode a caption for each image: [BOS][image tokens] -> text."""
        B = pixels.size(0)
        embed = self.lm.get_input_embeddings()
        img = self.encode_images(pixels)
        bos = embed(torch.full((B, 1), self.tokenizer.bos_token_id, device=self.device))
        emb = torch.cat([bos, img], dim=1)
        mask = torch.ones(emb.shape[:2], dtype=torch.long, device=self.device)
        kw = {}
        if stop_after_species:
            kw["stopping_criteria"] = StoppingCriteriaList([SpeciesDone(self.tokenizer)])
        with torch.autocast("cuda", enabled=False):
            out = self.lm.generate(inputs_embeds=emb, attention_mask=mask, max_new_tokens=max_new_tokens,
                                   do_sample=False, num_beams=1, top_k=None, top_p=None,
                                   eos_token_id=self.tokenizer.eos_token_id,
                                   pad_token_id=self.tokenizer.pad_token_id, **kw)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    def forward(self, pixels, input_ids, attn, labels, species_mask, eos_token_id):
        B, Lt = input_ids.shape
        img = self.encode_images(pixels)                                             # (B,49,H) fp32
        embed = self.lm.get_input_embeddings()

        bos = embed(torch.full((B, 1), self.tokenizer.bos_token_id, device=self.device))
        txt = embed(input_ids)
        emb = torch.cat([bos, img, txt], dim=1)
        mask = torch.cat([torch.ones(B, 1 + img.size(1), device=self.device, dtype=attn.dtype), attn], 1)

        # decoder runs in full fp32 (no autocast)
        with torch.autocast("cuda", enabled=False):
            out = self.lm(inputs_embeds=emb, attention_mask=mask, use_cache=False,
                          logits_to_keep=Lt + 1, return_dict=True)
        # kept positions = [last prefix token, text_0 ... text_{Lt-1}] -> logits[:, :-1] predicts text_0..text_{Lt-1}
        logits = out.logits[:, :-1].float()

        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                             reduction="none", ignore_index=-100, label_smoothing=0.1).view_as(labels)
        weights = torch.full_like(ce, 2.0)
        weights[labels == eos_token_id] = 5.0
        weights[species_mask] = 5.0
        valid = labels != -100
        return (ce * weights)[valid].sum() / weights[valid].sum()
