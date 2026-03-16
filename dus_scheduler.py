"""
DUS（Dilated Unmasking Scheduler）稀释调度算法。

将待解码的 mask 位置划分为 ceil(log2(N)) 步，
每步选取间隔最大的不相邻位置集合，使得每步内的 token
尽可能少受彼此影响，降低联合熵的增量。
"""

import math


def dilated_schedule(positions: list[int], base: int = 2) -> list[list[int]]:
    """
    对位置列表生成 DUS 稀释调度方案。

    参数：
        positions : 需要 unmask 的位置索引列表（顺序不限）
        base      : 稀释基数，默认 2（每步间距翻倍）

    返回：
        steps : list of list，每个子列表是该步要解码的位置集合。
                steps[0] 间距最大（最稀疏），steps[-1] 包含剩余位置。

    示例（8个位置，base=2）：
        positions = [2, 3, 4, 5, 6, 7, 8, 9]
        steps[0] = [2, 6]      stride=4
        steps[1] = [4, 8]      stride=4（偏移2）
        steps[2] = [3, 5, 7, 9] stride=2 的剩余
    """
    if not positions:
        return []

    sorted_pos = sorted(positions)
    n = len(sorted_pos)
    num_steps = max(1, math.ceil(math.log2(n + 1)))

    assigned = set()
    steps = []

    stride = 2 ** (num_steps - 1)
    while stride >= 1:
        step_positions = []
        for offset in range(stride):
            for idx in range(offset, n, stride * 2):
                p = sorted_pos[idx]
                if p not in assigned:
                    step_positions.append(p)
                    assigned.add(p)
        if step_positions:
            steps.append(sorted(step_positions))
        stride //= 2

    # 兜底：把还未分配的位置放到最后一步
    remaining = [p for p in sorted_pos if p not in assigned]
    if remaining:
        steps.append(remaining)

    return steps


def dus_decode(
    token_ids: list[int],
    target_positions: list[int],
    dllm,
    confidence_threshold: float = 0.5,
    temperature: float = 1.0,
    base: int = 2,
) -> list[int]:
    """
    对 target_positions 中的 mask 位置执行 DUS 调度解码。

    参数：
        token_ids            : 当前 token id 序列（含 [MASK]）
        target_positions     : 需要解码的位置列表
        dllm                 : DLLMModel 实例
        confidence_threshold : 低于此置信度则跳过（延后到兜底步）
        temperature          : 采样温度
        base                 : DUS 稀释基数

    返回：
        更新后的 token_ids（in-place 修改副本）
    """
    ids = list(token_ids)
    schedule = dilated_schedule(target_positions, base=base)
    deferred = []

    for step_positions in schedule:
        logits = dllm.forward(ids)
        for pos in step_positions:
            conf = dllm.get_confidence(logits, pos)
            if conf >= confidence_threshold:
                ids[pos] = dllm.sample(logits, pos, temperature)
            else:
                deferred.append(pos)

    # 兜底：仍为 [MASK] 的位置强制 argmax
    if deferred:
        logits = dllm.forward(ids)
        for pos in deferred:
            ids[pos] = dllm.argmax(logits, pos)

    return ids
