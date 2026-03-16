"""
主入口：用户故事分阶段生成流水线。

使用方式：
    python main.py --requirement "在线购物系统，用户可以浏览商品、下单、查看物流" --num 6

输出：
    每条 User Story 打印到控制台，同时保存到 output.json
"""

import argparse
import json

import config
from dllm_interface import DLLMModel
from graph_builder import build_graphs
from phase1 import generate_skeletons
from phase2 import generate_role_action
from phase4 import generate_benefits


def run_pipeline(requirement: str, num_stories: int) -> list[dict]:
    """
    执行四阶段生成流水线。

    参数：
        requirement  : 自然语言需求描述
        num_stories  : 希望生成的 User Story 数量

    返回：
        完整的 User Story 列表，每项包含 role / action / benefit 字段
    """
    print(f"\n需求：{requirement}")
    print(f"生成数量：{num_stories}\n")

    # 初始化 dLLM 模型
    print("加载 dLLM 模型...")
    dllm = DLLMModel(model_path=config.DLLM_MODEL_PATH, device="cuda")

    # ------------------------------------------------------------------ #
    # Phase 1：生成骨架
    # ------------------------------------------------------------------ #
    print("Phase 1：生成骨架...")
    skeletons = generate_skeletons(n=num_stories, dllm=dllm)
    print(f"  生成 {len(skeletons)} 个骨架，0 步完成。\n")

    # ------------------------------------------------------------------ #
    # Phase 2：生成角色和动作 + 验证 + 重 mask
    # ------------------------------------------------------------------ #
    print("Phase 2：生成角色和动作...")
    stories = generate_role_action(skeletons, requirement, dllm)

    passed = sum(1 for s in stories if not s["low_quality"])
    print(f"  通过验证：{passed}/{len(stories)}")
    for s in stories:
        mark = "✗(低质量)" if s["low_quality"] else "✓"
        print(f"  US{s['id']+1}: 作为 {s['role']} 我想要 {s['action']}  {mark}")
    print()

    # ------------------------------------------------------------------ #
    # Phase 3：构建用户故事关系图
    # ------------------------------------------------------------------ #
    print("Phase 3：构建关系图...")
    synonymy_graph, cooperation_graph, dependency_graph = build_graphs(stories)

    synonymy_edges   = list(synonymy_graph.edges(data=True))
    cooperation_edges = list(cooperation_graph.edges(data=True))
    dependency_edges  = list(dependency_graph.edges())

    print(f"  近义边：{[(e[0], e[1]) for e in synonymy_edges]}")
    print(f"  合作边：{[(e[0], e[1]) for e in cooperation_edges]}")
    print(f"  依赖边：{list(dependency_edges)}")
    print()

    # ------------------------------------------------------------------ #
    # Phase 4：根据关系图生成目的 + 验证 + 重 mask
    # ------------------------------------------------------------------ #
    print("Phase 4：生成目的（依赖 → 合作 → 近义 → 孤立）...")
    stories = generate_benefits(
        stories,
        synonymy_graph,
        cooperation_graph,
        dependency_graph,
        dllm,
    )
    print()

    # ------------------------------------------------------------------ #
    # 输出结果
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("生成结果：")
    print("=" * 60)
    for s in stories:
        mark = " [低质量]" if s["low_quality"] else ""
        print(f"US{s['id']+1}: 作为 {s['role']} 我想要 {s['action']} 以便于 {s['benefit']}{mark}")

    return stories


def main():
    parser = argparse.ArgumentParser(description="用户故事分阶段生成")
    parser.add_argument(
        "--requirement", "-r",
        type=str,
        required=True,
        help="自然语言需求描述，例如：'在线购物系统，用户可以浏览商品、下单、查看物流'",
    )
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=6,
        help="希望生成的 User Story 数量（默认 6）",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.json",
        help="输出 JSON 文件路径（默认 output.json）",
    )
    args = parser.parse_args()

    stories = run_pipeline(args.requirement, args.num)

    # 保存结果
    output = [
        {
            "id": s["id"] + 1,
            "user_story": f"作为{s['role']}，我想要{s['action']}，以便于{s['benefit']}",
            "role": s["role"],
            "action": s["action"],
            "benefit": s["benefit"],
            "low_quality": s["low_quality"],
        }
        for s in stories
    ]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
