"""
Phase 3：构建用户故事关系图。

输入：已确定角色和动作的故事列表
输出：三张关系图（近义 / 合作 / 依赖），用 networkx 表示

近义关系：角色相同 + 动作语义相似度 > 阈值
合作关系：两故事动作中有共享业务对象
依赖关系：一个动作的完成是另一个动作的前提
"""

from __future__ import annotations

import jieba.posseg as pseg
import networkx as nx

import config
from validator import _semantic_similarity, _llm_judge


# ------------------------------------------------------------------ #
# 辅助：业务对象提取 + 归一化
# ------------------------------------------------------------------ #

def _extract_business_objects(action: str) -> set[str]:
    """
    从动作文本中提取业务对象，归一化到标准词条。
    先用 jieba 提取名词，再查词表归一化。
    """
    nouns = {w for w, f in pseg.cut(action) if f in ("n", "nz", "nn", "nr")}
    normalized = set()
    for noun in nouns:
        for canonical, aliases in config.BUSINESS_OBJECTS.items():
            if noun in aliases or noun == canonical:
                normalized.add(canonical)
                break
        else:
            # 未命中词表的名词也保留，让规则有机会匹配
            normalized.add(noun)
    return normalized


def _shared_business_objects(action_a: str, action_b: str) -> list[str]:
    """返回两个动作共享的业务对象列表（归一化后的交集）。"""
    objs_a = _extract_business_objects(action_a)
    objs_b = _extract_business_objects(action_b)
    return sorted(objs_a & objs_b)


# ------------------------------------------------------------------ #
# 辅助：依赖关系判定
# ------------------------------------------------------------------ #

def _check_dependency_rules(action_a: str, action_b: str) -> bool:
    """
    用规则词表判定 action_a 是否依赖 action_b。
    遍历 DEPENDENCY_RULES：若 action_a 包含某 key，且 action_b 包含对应的某个 value → 依赖成立。
    """
    for key, prerequisites in config.DEPENDENCY_RULES.items():
        if key in action_a:
            if any(pre in action_b for pre in prerequisites):
                return True
    return False


def _check_dependency_llm(action_a: str, action_b: str) -> bool:
    """用 LLM 判定 action_a 是否依赖 action_b（兜底）。"""
    prompt = (
        f"在软件系统中，要完成动作"{action_a}"，是否必须先完成"{action_b}"？\n"
        f"只输出 是 或 否。"
    )
    return _llm_judge(prompt) == 1


# ------------------------------------------------------------------ #
# 主函数：构建三张关系图
# ------------------------------------------------------------------ #

def build_graphs(
    stories: list[dict],
    synonymy_threshold: float = config.SIM_SYNONYMY_THRESHOLD,
    use_llm_for_dependency: bool = True,
) -> tuple[nx.Graph, nx.Graph, nx.DiGraph]:
    """
    构建用户故事关系图。

    参数：
        stories : list of dict，每项包含
                  {"id": int, "role": str, "action": str, "token_ids": list[int], "regions": dict}
        synonymy_threshold  : 近义关系的动作相似度阈值
        use_llm_for_dependency : 是否在规则未命中时用 LLM 判定依赖

    返回：
        synonymy_graph  : nx.Graph  — 无向，边属性 {"sim": float}
        cooperation_graph : nx.Graph — 无向，边属性 {"shared": list[str]}
        dependency_graph  : nx.DiGraph — 有向 i→j 表示 i 依赖 j（j 先于 i）
    """
    synonymy_graph   = nx.Graph()
    cooperation_graph = nx.Graph()
    dependency_graph  = nx.DiGraph()

    n = len(stories)
    for s in stories:
        synonymy_graph.add_node(s["id"])
        cooperation_graph.add_node(s["id"])
        dependency_graph.add_node(s["id"])

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = stories[i], stories[j]

            # ---- 近义关系 ----
            if si["role"] == sj["role"]:
                sim = _semantic_similarity(si["action"], sj["action"])
                if sim >= synonymy_threshold:
                    synonymy_graph.add_edge(si["id"], sj["id"], sim=sim)

            # ---- 合作关系 ----
            shared = _shared_business_objects(si["action"], sj["action"])
            if shared:
                cooperation_graph.add_edge(si["id"], sj["id"], shared=shared)

            # ---- 依赖关系 ----
            # 检查 i 是否依赖 j
            if _check_dependency_rules(si["action"], sj["action"]):
                dependency_graph.add_edge(si["id"], sj["id"])
            elif use_llm_for_dependency and _check_dependency_llm(si["action"], sj["action"]):
                dependency_graph.add_edge(si["id"], sj["id"])

            # 检查 j 是否依赖 i
            if _check_dependency_rules(sj["action"], si["action"]):
                dependency_graph.add_edge(sj["id"], si["id"])
            elif use_llm_for_dependency and _check_dependency_llm(sj["action"], si["action"]):
                dependency_graph.add_edge(sj["id"], si["id"])

    return synonymy_graph, cooperation_graph, dependency_graph


def get_dependency_paths(dep_graph: nx.DiGraph) -> list[list[int]]:
    """
    从依赖图中提取所有简单路径（拓扑排序后的有向路径）。
    若存在环则忽略环中的回边（实际需求中依赖关系不应有环）。
    返回按拓扑顺序排列的路径列表，每条路径是 story id 的有序列表。
    """
    # 检测并去除环（正常需求中不应有循环依赖，此处做防御处理）
    dag = dep_graph.copy()
    while not nx.is_directed_acyclic_graph(dag):
        # 找到一条环并去掉其中一条边
        cycle = nx.find_cycle(dag)
        dag.remove_edge(*cycle[0])

    # 对弱连通分量分别提取路径
    paths = []
    for component in nx.weakly_connected_components(dag):
        sub = dag.subgraph(component).copy()
        # 找所有从入度为 0 的节点出发的路径
        roots = [n for n in sub.nodes if sub.in_degree(n) == 0]
        for root in roots:
            for target in sub.nodes:
                if root == target:
                    continue
                for path in nx.all_simple_paths(sub, root, target):
                    if len(path) >= 2:
                        paths.append(path)

    # 去重并按长度降序排列（长路径先处理，短路径可能是长路径的子集）
    unique_paths = list({tuple(p): p for p in paths}.values())
    unique_paths.sort(key=len, reverse=True)
    return unique_paths
