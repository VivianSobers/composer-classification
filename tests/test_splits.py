import numpy as np

from composer_clf.splits import grouped_split, random_split


def make(n_classes=3, groups_per_class=10, files_per_group=6):
    y, g = [], []
    for c in range(n_classes):
        for k in range(groups_per_class):
            y += [f"c{c}"] * files_per_group
            g += [f"c{c}:g{k}"] * files_per_group
    return np.array(y), np.array(g)


def test_grouped_split_keeps_groups_together():
    y, g = make()
    split = grouped_split(y, g, seed=0)
    for group in np.unique(g):
        assert len(set(split[g == group])) == 1


def test_grouped_split_has_every_class_everywhere_and_sizes():
    y, g = make()
    split = grouped_split(y, g, seed=1)
    for part, frac in [("train", 0.6), ("dev", 0.2), ("test", 0.2)]:
        assert set(y[split == part]) == set(y)
        assert abs((split == part).mean() - frac) < 0.1


def test_grouped_split_deterministic_and_seed_dependent():
    y, g = make()
    assert (grouped_split(y, g, 3) == grouped_split(y, g, 3)).all()
    assert (grouped_split(y, g, 3) != grouped_split(y, g, 4)).any()


def test_random_split_sizes_and_classes():
    y, _ = make()
    split = random_split(y, seed=0)
    for part in ("train", "dev", "test"):
        assert set(y[split == part]) == set(y)
    assert abs((split == "test").mean() - 0.2) < 0.05
