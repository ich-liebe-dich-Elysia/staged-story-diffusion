"""
Phase 2：生成角色和动作 + 验证 + 重 mask。

解码策略：置信度顺序解码（每步 unmask 当前置信度最高的位置），
不使用 DUS，因为此阶段 mask 数量少（~8 个）且质量优先。

验证后若有错误，按字段粒度重 mask 并重新生成（最多 MAX_RETRY 次）。
"""

from __future__ import annotations

import config
from dllm_interface import DLLMModel
from validator import validate_role_action


def _confidence_ordered_decode(
    token_ids: list[int],
    target_positions: list[int],
    dllm: DLLMModel,
    temperature: float = 1.0,
    confidence_threshold: float = config.CONFIDENCE_THRESHOLD_PHASE2,
) -> list[int]:
    """
    置信度顺序解码：每步选当前置信度最高的位置 unmask。

    若最高置信度仍低于阈值，则强制 argmax 解码（避免死循环）。
    """
    ids = list(token_ids)
    remaining = list(target_positions)

    while remaining:
        logits = dllm.forward(ids)
        # 找出 remaining 中置信度最高的位置
        best_pos = max(remaining, key=lambda p: dllm.get_confidence(logits, p))
        conf = dllm.get_confidence(logits, best_pos)

        if conf >= confidence_threshold:
            ids[best_pos] = dllm.sample(logits, best_pos, temperature)
        else:
            # 置信度不足，强制解码（兜底）
            ids[best_pos] = dllm.argmax(logits, best_pos)

        remaining.remove(best_pos)

    return ids


def _remask_by_errors(
    token_ids: list[int],
    regions: dict,
    errors: list[dict],
    mask_token_id: int,
) -> tuple[list[int], list[int]]:
    """
    根据错误列表精确重 mask 对应字段，返回更新后的 token_ids 和重 mask 的位置列表。

    错误字段映射：
      "role"   → 重 mask 角色区
      "action" → 重 mask 动作区
    """
    ids = list(token_ids)
    remask_positions = []

    error_fields = {e["field"] for e in errors}

    if "role" in error_fields:
        rs, re = regions["role"]
        for p in range(rs, re):
            ids[p] = mask_token_id
            remask_positions.append(p)

    if "action" in error_fields:
        rs, re = regions["action"]
        for p in range(rs, re):
            ids[p] = mask_token_id
            remask_positions.append(p)

    return ids, remask_positions


def generate_role_action(
    skeletons: list[dict],
    requirement: str,
    dllm: DLLMModel,
) -> list[dict]:
    """
    对每个骨架生成角色和动作，验证后重 mask 重试。

    参数：
        skeletons   : Phase 1 输出的骨架列表
        requirement : 原始需求文本
        dllm        : DLLMModel 实例

    返回：
        stories : 同结构列表，每项的 token_ids 中角色区和动作区已填充，
                  目的区仍为 [MASK]。low_quality=True 表示 3 次重试均未通过。
    """
    stories = []

    for skeleton in skeletons:
        token_ids = list(skeleton["token_ids"])
        regions   = skeleton["regions"]

        role_start, role_end     = regions["role"]
        action_start, action_end = regions["action"]
        target_positions = list(range(role_start, role_end)) + list(range(action_start, action_end))

        # 初始解码
        token_ids = _confidence_ordered_decode(
            token_ids, target_positions, dllm,
            temperature=1.0,
        )

        low_quality = False
        for retry_idx in range(config.MAX_RETRY):
            role   = dllm.decode_region(token_ids, role_start, role_end)
            action = dllm.decode_region(token_ids, action_start, action_end)
            errors = validate_role_action(role, action, requirement)

            if not errors:
                break

            # 按字段重 mask
            temperature = config.RETRY_TEMPERATURES[retry_idx]
            token_ids, remask_pos = _remask_by_errors(
                token_ids, regions, errors, dllm.mask_token_id
            )
            # 重新解码（降温）
            token_ids = _confidence_ordered_decode(
                token_ids, remask_pos, dllm, temperature=temperature,
            )
        else:
            # 3 次重试均失败
            low_quality = True

        role   = dllm.decode_region(token_ids, role_start, role_end)
        action = dllm.decode_region(token_ids, action_start, action_end)

        stories.append({
            **skeleton,
            "token_ids":   token_ids,
            "role":        role,
            "action":      action,
            "low_quality": low_quality,
        })

    return stories
