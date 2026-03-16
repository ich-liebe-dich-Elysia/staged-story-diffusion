"""
dLLM 模型接口封装。

封装离散扩散语言模型的前向推理，对外提供统一接口：
- forward(input_ids) → logits
- get_confidence(logits, pos) → float
- sample(logits, pos, temperature) → token_id
- encode / decode 文本
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM


class DLLMModel:
    """
    离散扩散语言模型封装。
    实际使用时替换为对应的 dLLM 实现（如 SEDD / MDLM 等）。
    此处以 MaskedLM 接口为抽象层，行为与 dLLM 推理时一致：
    输入含 [MASK] 的 token 序列，输出每个位置的 logits。
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

        self.mask_token_id = self.tokenizer.mask_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    @torch.no_grad()
    def forward(self, input_ids: list[int]) -> torch.Tensor:
        """
        对含 [MASK] 的序列做一次前向推理。
        返回 logits，shape = (seq_len, vocab_size)。
        """
        ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = (ids != self.pad_token_id).long()
        outputs = self.model(input_ids=ids, attention_mask=attention_mask)
        return outputs.logits[0]  # (seq_len, vocab_size)

    def get_confidence(self, logits: torch.Tensor, pos: int) -> float:
        """返回某位置 logits 对应的最大 softmax 概率（作为置信度）。"""
        probs = F.softmax(logits[pos], dim=-1)
        return probs.max().item()

    def sample(self, logits: torch.Tensor, pos: int, temperature: float = 1.0) -> int:
        """按温度从某位置的 logits 采样一个 token id。"""
        if temperature <= 0:
            return logits[pos].argmax().item()
        scaled = logits[pos] / temperature
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def argmax(self, logits: torch.Tensor, pos: int) -> int:
        """贪心取最高概率 token。"""
        return logits[pos].argmax().item()

    # ------------------------------------------------------------------ #
    # 文本 <-> token id 转换
    # ------------------------------------------------------------------ #

    def encode_template(self, role_len: int, action_len: int, benefit_len: int) -> tuple[list[int], dict]:
        """
        生成骨架的 token id 序列，以及各区域的位置索引。

        骨架格式：
          [CLS] 作 为 [MASK]*role_len 我 想 要 [MASK]*action_len 以 便 于 [MASK]*benefit_len [SEP]

        返回：
          token_ids : list[int]
          regions   : {"role": (start, end), "action": (start, end), "benefit": (start, end)}
                      均为左闭右开区间，索引对应 token_ids 中的位置
        """
        prefix_as    = self.tokenizer.encode("作为", add_special_tokens=False)
        prefix_want  = self.tokenizer.encode("我想要", add_special_tokens=False)
        prefix_benef = self.tokenizer.encode("以便于", add_special_tokens=False)

        ids = [self.cls_token_id]

        ids += prefix_as
        role_start = len(ids)
        ids += [self.mask_token_id] * role_len
        role_end = len(ids)

        ids += prefix_want
        action_start = len(ids)
        ids += [self.mask_token_id] * action_len
        action_end = len(ids)

        ids += prefix_benef
        benefit_start = len(ids)
        ids += [self.mask_token_id] * benefit_len
        benefit_end = len(ids)

        ids += [self.sep_token_id]

        regions = {
            "role":    (role_start,    role_end),
            "action":  (action_start,  action_end),
            "benefit": (benefit_start, benefit_end),
        }
        return ids, regions

    def decode_region(self, token_ids: list[int], start: int, end: int) -> str:
        """将某区间的 token ids 解码为文本，去除特殊符号。"""
        ids = token_ids[start:end]
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def concat_with_sep(self, sequences: list[list[int]]) -> tuple[list[int], list[int]]:
        """
        将多个 token id 序列用 [SEP] 拼接（去掉各自首尾的 [CLS]/[SEP]），
        并重新添加首尾特殊符号。

        返回：
          combined   : 拼接后的 token ids
          boundaries : 每个故事在 combined 中的起始索引（不含 [CLS]）
        """
        combined = [self.cls_token_id]
        boundaries = []
        for i, seq in enumerate(sequences):
            # 去掉首尾特殊 token
            inner = [t for t in seq if t not in (self.cls_token_id, self.sep_token_id)]
            boundaries.append(len(combined))
            combined += inner
            if i < len(sequences) - 1:
                combined.append(self.sep_token_id)
        combined.append(self.sep_token_id)
        return combined, boundaries
