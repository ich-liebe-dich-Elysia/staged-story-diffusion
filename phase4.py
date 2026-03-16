"""
Phase 4：根据关系图生成目的区 + 验证 + 重 mask。

处理顺序：依赖（最强约束）→ 合作 → 近义 → 孤立故事（最弱约束）

已处理集合防止重复解码；低优先级阶段将高优先级的已处理故事作为上下文（只读）。
"""

from __future__ import annotations

import networkx as nx

import config
from dllm_interface import DLLMModel
from dus_scheduler import dus_decode
from validator import (
    validate_benefit_quality,
    validate_benefit_dependency,
    validate_benefit_cooperation,
    validate_benefit_synonymy,
)


# ------------------------------------------------------------------ #
# 辅助：目的区重 mask
# ------------------------------------------------------------------ #

def _remask_benefit(
    token_ids: list[int],
    regions: dict,
    mask_token_id: int,
) -> tuple[list[int], list[int]]:
    """将目的区全部重 mask，返回更新后的 ids 和重 mask 位置。"""
    ids = list(token_ids)
    bs, be = regions["benefit"]
    positions = list(range(bs, be))
    for p in positions:
        ids[p] = mask_token_id
    return ids, positions


# ------------------------------------------------------------------ #
# 4c：依赖关系生成
# ------------------------------------------------------------------ #

def generate_benefit_dependency(
    path: list[int],
    stories: dict[int, dict],
    processed: set[int],
    dllm: DLLMModel,
) -> None:
    """
    沿依赖路径（拓扑顺序）逐故事生成目的。
    前驱故事的完整 token 序列作为 prefix 拼接到当前故事前。

    修改 stories[id]["token_ids"] 和 stories[id]["benefit"]（in-place）。
    """
    context_ids: list[list[int]] = []  # 累积的前驱故事 token 序列

    for story_id in path:
        story = stories[story_id]

        if story_id in processed:
            # 已处理：直接作为上下文追加
            context_ids.append(story["token_ids"])
            continue

        # 构造 full_seq = context + [SEP] + current_story
        if context_ids:
            full_ids, _ = dllm.concat_with_sep(context_ids + [story["token_ids"]])
            # 计算目的区在 full_seq 中的绝对位置
            offset = len(full_ids) - len(story["token_ids"])
            bs, be = story["regions"]["benefit"]
            benefit_positions = [p + offset for p in range(bs, be)]
        else:
            full_ids = list(story["token_ids"])
            bs, be = story["regions"]["benefit"]
            benefit_positions = list(range(bs, be))

        # 获取前置故事信息（用于跨故事验证）
        prev_story = stories[path[path.index(story_id) - 1]] if path.index(story_id) > 0 else None

        for retry_idx in range(config.MAX_RETRY):
            temperature = 1.0 if retry_idx == 0 else config.RETRY_TEMPERATURES[retry_idx - 1]
            full_ids = dus_decode(
                full_ids, benefit_positions, dllm,
                confidence_threshold=config.DUS_CONFIDENCE_THRESHOLD,
                temperature=temperature,
            )

            # 将解码结果写回 story["token_ids"]
            if context_ids:
                offset = len(full_ids) - len(story["token_ids"])
                bs, be = story["regions"]["benefit"]
                for p in range(bs, be):
                    story["token_ids"][p] = full_ids[p + offset]
            else:
                story["token_ids"] = list(full_ids)

            benefit = dllm.decode_region(story["token_ids"], *story["regions"]["benefit"])
            action  = story["action"]

            # 第一层验证：单故事质量
            errors = validate_benefit_quality(benefit, action)

            # 第二层验证：依赖关系（与前置故事不矛盾）
            if not errors and prev_story and prev_story.get("benefit"):
                errors += validate_benefit_dependency(
                    benefit, prev_story["benefit"],
                    action, prev_story["action"],
                )

            if not errors:
                break

            # 重 mask 目的区，下一轮重新生成
            story["token_ids"], benefit_positions_local = _remask_benefit(
                story["token_ids"], story["regions"], dllm.mask_token_id
            )
            if context_ids:
                offset = len(full_ids) - len(story["token_ids"])
                benefit_positions = [p + offset for p in benefit_positions_local]
                # 同步 full_ids 中的目的区
                for p_local, p_full in zip(benefit_positions_local, benefit_positions):
                    full_ids[p_full] = dllm.mask_token_id
            else:
                full_ids = list(story["token_ids"])
                benefit_positions = benefit_positions_local
        else:
            story["low_quality"] = True

        story["benefit"] = dllm.decode_region(story["token_ids"], *story["regions"]["benefit"])
        processed.add(story_id)
        context_ids.append(story["token_ids"])


# ------------------------------------------------------------------ #
# 4b：合作关系生成
# ------------------------------------------------------------------ #

def generate_benefit_cooperation(
    component: list[int],
    stories: dict[int, dict],
    processed: set[int],
    dllm: DLLMModel,
) -> None:
    """
    将分量内故事拼接，只对未处理故事的目的区做跨故事 DUS 解码。
    已处理故事保持原文作为上下文。
    """
    unprocessed = [sid for sid in component if sid not in processed]
    context_done = [sid for sid in component if sid in processed]

    if not unprocessed:
        return

    # 构造 combined 序列：context_done 在前（目的已填充），unprocessed 在后（目的仍为 [MASK]）
    ordered = context_done + unprocessed
    seqs = [stories[sid]["token_ids"] for sid in ordered]
    combined, boundaries = dllm.concat_with_sep(seqs)

    # 计算 unprocessed 故事的目的区在 combined 中的绝对位置
    num_context = len(context_done)
    benefit_masks: list[int] = []
    unprocessed_benefit_ranges: dict[int, list[int]] = {}  # story_id → absolute positions

    for local_idx, sid in enumerate(unprocessed):
        global_idx = num_context + local_idx
        boundary_offset = boundaries[global_idx]
        bs, be = stories[sid]["regions"]["benefit"]
        # 在各自 token_ids 中的偏移要加上 boundary_offset 减去 [CLS] 偏移
        # concat_with_sep 去掉了 [CLS]/[SEP]，boundary 从 1 开始（[CLS] 占位 0）
        # boundary_offset 已是在 combined 中的实际起始位置（不含 [CLS]）
        # 需要加 1（因为 combined[0] = [CLS]）
        inner_start = boundaries[global_idx]
        # 在原始 token_ids 中，[CLS] 在位置 0，所以 bs 是相对于 [CLS] 后
        # inner 序列去掉了 [CLS]，所以 bs - 1 是在 inner 中的位置
        abs_positions = [inner_start + (bs - 1) + k for k in range(be - bs)]
        benefit_masks.extend(abs_positions)
        unprocessed_benefit_ranges[sid] = abs_positions

    # 获取合作关系中的共享业务对象（用于跨故事验证）
    all_shared: list[str] = []
    # 从 cooperation_graph 边属性获取，此处简化为提取所有动作的名词
    for sid in unprocessed:
        from graph_builder import _extract_business_objects
        all_shared.extend(_extract_business_objects(stories[sid]["action"]))
    all_shared = list(set(all_shared))

    for retry_idx in range(config.MAX_RETRY):
        temperature = 1.0 if retry_idx == 0 else config.RETRY_TEMPERATURES[retry_idx - 1]
        combined = dus_decode(
            combined, benefit_masks, dllm,
            confidence_threshold=config.DUS_CONFIDENCE_THRESHOLD,
            temperature=temperature,
        )

        # 将解码结果写回各故事 token_ids
        for sid in unprocessed:
            abs_positions = unprocessed_benefit_ranges[sid]
            bs, be = stories[sid]["regions"]["benefit"]
            for k, abs_p in enumerate(abs_positions):
                stories[sid]["token_ids"][bs + k] = combined[abs_p]
            stories[sid]["benefit"] = dllm.decode_region(
                stories[sid]["token_ids"], bs, be
            )

        # 第一层验证：各未处理故事单独检查
        all_passed = True
        failed_positions: list[int] = []
        for sid in unprocessed:
            benefit = stories[sid]["benefit"]
            action  = stories[sid]["action"]
            errors  = validate_benefit_quality(benefit, action)
            if errors:
                all_passed = False
                bs, be = stories[sid]["regions"]["benefit"]
                for p in unprocessed_benefit_ranges[sid]:
                    combined[p] = dllm.mask_token_id
                stories[sid]["token_ids"], _ = _remask_benefit(
                    stories[sid]["token_ids"], stories[sid]["regions"], dllm.mask_token_id
                )
                failed_positions.extend(unprocessed_benefit_ranges[sid])

        # 第二层验证：合作关系（分量内至少一个故事引用共享业务对象）
        if all_passed and all_shared:
            benefits = [stories[sid]["benefit"] for sid in component]
            cross_errors = validate_benefit_cooperation(benefits, all_shared)
            if cross_errors:
                all_passed = False
                # 重 mask 所有未处理故事
                for sid in unprocessed:
                    combined_positions = unprocessed_benefit_ranges[sid]
                    for p in combined_positions:
                        combined[p] = dllm.mask_token_id
                    stories[sid]["token_ids"], _ = _remask_benefit(
                        stories[sid]["token_ids"], stories[sid]["regions"], dllm.mask_token_id
                    )
                failed_positions = benefit_masks

        if all_passed:
            break

        benefit_masks = failed_positions

    else:
        for sid in unprocessed:
            stories[sid]["low_quality"] = True

    for sid in unprocessed:
        processed.add(sid)


# ------------------------------------------------------------------ #
# 4a：近义关系生成
# ------------------------------------------------------------------ #

def generate_benefit_synonymy(
    component: list[int],
    stories: dict[int, dict],
    processed: set[int],
    dllm: DLLMModel,
) -> None:
    """
    以锚点故事为上下文，顺序生成其余未处理故事的目的。
    若分量内已有处理过的故事，直接用它作锚点；否则先生成第一个故事作锚点。
    """
    unprocessed = [sid for sid in component if sid not in processed]
    context_done = [sid for sid in component if sid in processed]

    if not unprocessed:
        return

    # 确定锚点
    if context_done:
        anchor_id = context_done[0]
        # 锚点已有目的，直接使用
    else:
        anchor_id = unprocessed[0]
        unprocessed = unprocessed[1:]

        # 单独生成锚点的目的区
        bs, be = stories[anchor_id]["regions"]["benefit"]
        benefit_positions = list(range(bs, be))

        for retry_idx in range(config.MAX_RETRY):
            temperature = 1.0 if retry_idx == 0 else config.RETRY_TEMPERATURES[retry_idx - 1]
            stories[anchor_id]["token_ids"] = dus_decode(
                stories[anchor_id]["token_ids"], benefit_positions, dllm,
                confidence_threshold=config.DUS_CONFIDENCE_THRESHOLD,
                temperature=temperature,
            )
            anchor_benefit = dllm.decode_region(stories[anchor_id]["token_ids"], bs, be)
            errors = validate_benefit_quality(anchor_benefit, stories[anchor_id]["action"])
            if not errors:
                break
            stories[anchor_id]["token_ids"], benefit_positions = _remask_benefit(
                stories[anchor_id]["token_ids"], stories[anchor_id]["regions"], dllm.mask_token_id
            )
        else:
            stories[anchor_id]["low_quality"] = True

        stories[anchor_id]["benefit"] = dllm.decode_region(
            stories[anchor_id]["token_ids"], *stories[anchor_id]["regions"]["benefit"]
        )
        processed.add(anchor_id)

    anchor_benefit = stories[anchor_id]["benefit"]
    anchor_ids = stories[anchor_id]["token_ids"]

    # 以锚点为上下文，逐个生成其余未处理故事
    for sid in unprocessed:
        story = stories[sid]
        # 将锚点故事作为 prefix 拼接
        combined, boundaries = dllm.concat_with_sep([anchor_ids, story["token_ids"]])
        offset = boundaries[1]  # story 在 combined 中的起始（不含 [CLS]）
        bs, be = story["regions"]["benefit"]
        benefit_positions = [offset + (bs - 1) + k for k in range(be - bs)]

        for retry_idx in range(config.MAX_RETRY):
            temperature = 1.0 if retry_idx == 0 else config.RETRY_TEMPERATURES[retry_idx - 1]
            combined = dus_decode(
                combined, benefit_positions, dllm,
                confidence_threshold=config.DUS_CONFIDENCE_THRESHOLD,
                temperature=temperature,
            )

            # 写回 story["token_ids"]
            for k, abs_p in enumerate(benefit_positions):
                story["token_ids"][bs + k] = combined[abs_p]
            benefit = dllm.decode_region(story["token_ids"], bs, be)
            action  = story["action"]

            # 第一层验证
            errors = validate_benefit_quality(benefit, action)

            # 第二层验证：近义相似度区间
            if not errors:
                errors += validate_benefit_synonymy(benefit, anchor_benefit)

            if not errors:
                break

            # 重 mask
            story["token_ids"], local_positions = _remask_benefit(
                story["token_ids"], story["regions"], dllm.mask_token_id
            )
            for p_full in benefit_positions:
                combined[p_full] = dllm.mask_token_id
        else:
            story["low_quality"] = True

        story["benefit"] = dllm.decode_region(story["token_ids"], bs, be)
        processed.add(sid)


# ------------------------------------------------------------------ #
# 4d：孤立故事生成
# ------------------------------------------------------------------ #

def generate_benefit_isolated(
    story: dict,
    dllm: DLLMModel,
) -> None:
    """对孤立故事单独 DUS 生成目的。"""
    bs, be = story["regions"]["benefit"]
    benefit_positions = list(range(bs, be))

    for retry_idx in range(config.MAX_RETRY):
        temperature = 1.0 if retry_idx == 0 else config.RETRY_TEMPERATURES[retry_idx - 1]
        story["token_ids"] = dus_decode(
            story["token_ids"], benefit_positions, dllm,
            confidence_threshold=config.DUS_CONFIDENCE_THRESHOLD,
            temperature=temperature,
        )
        benefit = dllm.decode_region(story["token_ids"], bs, be)
        errors  = validate_benefit_quality(benefit, story["action"])
        if not errors:
            break
        story["token_ids"], benefit_positions = _remask_benefit(
            story["token_ids"], story["regions"], dllm.mask_token_id
        )
    else:
        story["low_quality"] = True

    story["benefit"] = dllm.decode_region(story["token_ids"], bs, be)


# ------------------------------------------------------------------ #
# Phase 4 主函数
# ------------------------------------------------------------------ #

def generate_benefits(
    stories_list: list[dict],
    synonymy_graph: nx.Graph,
    cooperation_graph: nx.Graph,
    dependency_graph: nx.DiGraph,
    dllm: DLLMModel,
) -> list[dict]:
    """
    按 依赖 → 合作 → 近义 → 孤立 顺序生成所有故事的目的区。

    参数：
        stories_list     : Phase 2 输出的故事列表
        synonymy_graph   : 近义关系图
        cooperation_graph: 合作关系图
        dependency_graph : 依赖关系图
        dllm             : DLLMModel 实例

    返回：
        stories_list（in-place 修改，benefit 字段已填充）
    """
    from graph_builder import get_dependency_paths

    # 转为 id→dict 字典方便按 id 访问
    stories: dict[int, dict] = {s["id"]: s for s in stories_list}
    processed: set[int] = set()

    # ---- 4c：依赖关系（最强约束，最先处理）----
    dep_paths = get_dependency_paths(dependency_graph)
    for path in dep_paths:
        unprocessed_in_path = [sid for sid in path if sid not in processed]
        if not unprocessed_in_path:
            continue
        generate_benefit_dependency(path, stories, processed, dllm)

    # ---- 4b：合作关系 ----
    for component in nx.connected_components(cooperation_graph):
        component = list(component)
        unprocessed = [sid for sid in component if sid not in processed]
        if not unprocessed:
            continue
        generate_benefit_cooperation(component, stories, processed, dllm)

    # ---- 4a：近义关系（最弱约束，最后处理）----
    for component in nx.connected_components(synonymy_graph):
        component = list(component)
        unprocessed = [sid for sid in component if sid not in processed]
        if not unprocessed:
            continue
        generate_benefit_synonymy(component, stories, processed, dllm)

    # ---- 4d：孤立故事 ----
    for sid, story in stories.items():
        if sid not in processed:
            generate_benefit_isolated(story, dllm)
            processed.add(sid)

    return stories_list
