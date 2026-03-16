"""
验证模块：Phase 2 角色/动作验证 + Phase 4 目的验证（两层）。

工具栈：
  - jieba.posseg     : 词性标注（角色/动词检测）
  - sentence_transformers : 语义相似度
  - OpenAI-compatible LLM : 复杂语义判断（角色-动作一致性、因果关系）
"""

from __future__ import annotations

import jieba.posseg as pseg
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config


# ------------------------------------------------------------------ #
# 单例：懒加载模型
# ------------------------------------------------------------------ #

_sim_model: SentenceTransformer | None = None
_llm_client: OpenAI | None = None


def get_sim_model() -> SentenceTransformer:
    global _sim_model
    if _sim_model is None:
        _sim_model = SentenceTransformer(config.SENTENCE_MODEL_NAME)
    return _sim_model


def get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_API_BASE,
        )
    return _llm_client


# ------------------------------------------------------------------ #
# 基础工具函数
# ------------------------------------------------------------------ #

def _encode(text: str) -> np.ndarray:
    return get_sim_model().encode([text])[0]


def _semantic_similarity(a: str, b: str) -> float:
    va = _encode(a).reshape(1, -1)
    vb = _encode(b).reshape(1, -1)
    return float(cosine_similarity(va, vb)[0][0])


def _llm_judge(prompt: str) -> int:
    """调用 LLM 进行 0/1 判断，返回 0 或 1。"""
    client = get_llm_client()
    resp = client.chat.completions.create(
        model=config.LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4,
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    return 1 if raw.startswith("1") or raw.startswith("是") else 0


def _has_verb(text: str) -> bool:
    """判断文本中是否包含动词（v / vn / vd）。"""
    for _, flag in pseg.cut(text):
        if flag.startswith("v"):
            return True
    return False


def _is_valid_role(role: str) -> str | None:
    """
    验证角色是否为人/组织类词。
    返回 None 表示合法；返回错误描述字符串表示非法。
    """
    if not role.strip():
        return "角色缺失"
    # 白名单快速通过
    if any(w in role for w in config.ROLE_WHITELIST):
        return None
    # 黑名单直接拒绝
    if any(w in role for w in config.ROLE_BLACKLIST):
        return "角色无效（技术术语）"
    # jieba 词性兜底：含人名/专名词性则认为合法
    pos_tags = [flag for _, flag in pseg.cut(role)]
    if any(f in ("nr", "nz", "n") for f in pos_tags):
        return None
    return "角色无效（非人/组织）"


# ------------------------------------------------------------------ #
# Phase 2 验证：角色 + 动作
# ------------------------------------------------------------------ #

def validate_role_action(
    role: str,
    action: str,
    requirement: str,
) -> list[dict]:
    """
    验证角色和动作的正确性。

    返回错误列表，每项为 {"field": "role"|"action", "reason": str}。
    空列表表示通过。
    """
    errors = []

    # 1. 角色验证
    role_err = _is_valid_role(role)
    if role_err:
        errors.append({"field": "role", "reason": role_err})

    # 2. 动作验证
    if not action.strip():
        errors.append({"field": "action", "reason": "动作缺失"})
        return errors

    if not _has_verb(action):
        errors.append({"field": "action", "reason": "动作无动词"})

    sim = _semantic_similarity(action, requirement)
    if sim < config.SIM_ACTION_REQ_MIN:
        errors.append({"field": "action", "reason": f"动作与需求无关（相似度={sim:.2f}）"})

    # 3. 角色-动作一致性（LLM 判断）
    if not errors:  # 只有前面都通过才做 LLM 判断，节省调用
        prompt = (
            f"用户故事中，角色是"{role}"，动作是"{action}"。\n"
            f"请判断这个角色做这个动作是否合理（1=合理，0=不合理），只输出数字。"
        )
        if _llm_judge(prompt) == 0:
            errors.append({"field": "action", "reason": "角色-动作不一致"})

    return errors


# ------------------------------------------------------------------ #
# Phase 4 验证：目的（第一层：单故事质量）
# ------------------------------------------------------------------ #

def validate_benefit_quality(benefit: str, action: str) -> list[dict]:
    """
    单故事目的质量检查（所有关系通用）。
    返回错误列表，空列表表示通过。
    """
    errors = []

    if not benefit.strip():
        errors.append({"field": "benefit", "reason": "目的缺失"})
        return errors

    # 与动作相似度过高 → 重复
    sim = _semantic_similarity(benefit, action)
    if sim > config.SIM_BENEFIT_ACTION_MAX:
        errors.append({"field": "benefit", "reason": f"目的与动作重复（相似度={sim:.2f}）"})

    # 包含技术术语
    for term in config.TECHNICAL_TERMS:
        if term in benefit:
            errors.append({"field": "benefit", "reason": f"目的包含技术术语（{term}）"})
            break

    # 因果关系（LLM 判断）
    if not errors:
        prompt = (
            f"用户故事：我想要"{action}"，以便于"{benefit}"。\n"
            f"请判断"以便于"后面的目的是否是该动作的合理结果（1=是，0=否），只输出数字。"
        )
        if _llm_judge(prompt) == 0:
            errors.append({"field": "benefit", "reason": "目的与动作无因果关系"})

    return errors


# ------------------------------------------------------------------ #
# Phase 4 验证：目的（第二层：跨故事关系一致性）
# ------------------------------------------------------------------ #

def validate_benefit_dependency(
    benefit: str,
    prerequisite_benefit: str,
    action: str,
    prerequisite_action: str,
) -> list[dict]:
    """
    依赖关系跨故事验证：后置故事的目的不应与前置故事的目的矛盾。
    """
    errors = []
    prompt = (
        f"在软件系统中，故事A的动作是"{prerequisite_action}"，目的是"{prerequisite_benefit}"。\n"
        f"故事B的动作是"{action}"，目的是"{benefit}"。\n"
        f"故事B依赖故事A（先完成A才能做B）。\n"
        f"请判断故事B的目的在故事A已完成的前提下是否合理（1=合理，0=不合理），只输出数字。"
    )
    if _llm_judge(prompt) == 0:
        errors.append({"field": "benefit", "reason": "目的与前置故事矛盾（依赖关系冲突）"})
    return errors


def validate_benefit_cooperation(
    benefits: list[str],
    shared_objects: list[str],
) -> list[dict]:
    """
    合作关系跨故事验证：分量内至少一个故事的目的应引用共享业务对象。
    """
    errors = []
    all_text = " ".join(benefits)
    if not any(obj in all_text for obj in shared_objects):
        errors.append({
            "field": "benefit",
            "reason": f"合作分量中无故事目的引用共享对象（{shared_objects}）",
        })
    return errors


def validate_benefit_synonymy(
    benefit: str,
    anchor_benefit: str,
) -> list[dict]:
    """
    近义关系跨故事验证：目的与锚点的语义相似度应在 [min, max] 区间内。
    """
    errors = []
    sim = _semantic_similarity(benefit, anchor_benefit)
    if sim < config.SIM_BENEFIT_SYNONYMY_MIN:
        errors.append({"field": "benefit", "reason": f"目的主题漂移（与锚点相似度={sim:.2f}，过低）"})
    elif sim > config.SIM_BENEFIT_SYNONYMY_MAX:
        errors.append({"field": "benefit", "reason": f"目的与锚点冗余（与锚点相似度={sim:.2f}，过高）"})
    return errors
