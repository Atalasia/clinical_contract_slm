from pathlib import Path

from cpg2.tree import load_tree


ROOT = Path(__file__).resolve().parents[1]


def demo_tree():
    return load_tree(ROOT / "examples" / "mechanical_tree.json")
