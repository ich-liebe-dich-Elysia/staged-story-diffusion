"""
Phase 1：生成骨架。

直接将模板关键词写入序列，mask 区域留空，0 步完成。
"""

from dllm_interface import DLLMModel
import config


def generate_skeletons(
    n: int,
    dllm: DLLMModel,
) -> list[dict]:
    """
    生成 n 个骨架序列。

    每个骨架包含：
      token_ids : list[int]  — 含 [MASK] 的完整 token id 序列
      regions   : dict       — {"role": (s,e), "action": (s,e), "benefit": (s,e)}

    骨架格式：
      [CLS] 作为 [M][M] 我想要 [M][M][M][M][M][M] 以便于 [M][M][M][M][M][M] [SEP]
    """
    skeletons = []
    for i in range(n):
        token_ids, regions = dllm.encode_template(
            role_len=config.ROLE_LEN,
            action_len=config.ACTION_LEN,
            benefit_len=config.BENEFIT_LEN,
        )
        skeletons.append({
            "id": i,
            "token_ids": token_ids,
            "regions": regions,
            "role": "",
            "action": "",
            "benefit": "",
            "valid": True,
            "low_quality": False,
        })
    return skeletons
