import math
import random
import unittest

from datasketch import (
    ClassSplitStats,
    NearDuplicateGrouping,
    TrainValidationSplit,
    group_near_duplicates,
    split_train_validation,
)
from datasketch.split import _solve_validation_groups


def _brute_force(sizes, counts, total_target, class_targets):
    """Exhaustive reference for the lexicographic objective (small inputs only)."""
    best = None
    for mask in range(1 << len(sizes)):
        chosen = [index for index in range(len(sizes)) if mask >> index & 1]
        rows = sum(sizes[index] for index in chosen)
        key = (
            abs(rows - total_target),
            sum(
                abs(sum(counts[index][class_index] for index in chosen) - class_targets[class_index])
                for class_index in range(len(class_targets))
            ),
        )
        if best is None or key < best:
            best = key
    return best


class TestGroupNearDuplicates(unittest.TestCase):
    def test_transitive_chain_forms_one_group(self):
        # Each consecutive pair is above the threshold; the ends are not.
        sets = [[f"t{i}" for i in range(i, i + 10)] for i in range(3)]
        sets[0] = [f"t{i}" for i in range(0, 10)]
        sets[1] = [f"t{i}" for i in range(4, 14)]
        sets[2] = [f"t{i}" for i in range(8, 18)]
        grouping = group_near_duplicates(sets, jaccard_threshold=0.4, lsh_threshold=0.35)
        self.assertIsInstance(grouping, NearDuplicateGrouping)
        self.assertEqual(grouping.groups, [[0, 1, 2]])
        self.assertEqual(grouping.group_of, [0, 0, 0])
        self.assertEqual(grouping.num_non_singleton_groups, 1)
        self.assertEqual([pair[:2] for pair in grouping.verified_pairs], [(0, 1), (1, 2)])

    def test_exact_verification_rejects_lsh_false_positives(self):
        # Sliding windows share 9/11 tokens (Jaccard ~0.82): a low LSH
        # threshold recalls them, but the exact check at 0.9 must not link them.
        sets = [[f"x{j}" for j in range(i, i + 10)] for i in range(10)]
        grouping = group_near_duplicates(sets, jaccard_threshold=0.9, lsh_threshold=0.5, num_perm=64)
        self.assertGreater(grouping.candidate_pairs, 0)
        self.assertEqual(grouping.verified_pairs, [])
        self.assertEqual(grouping.rejected_pairs, grouping.candidate_pairs)
        self.assertTrue(all(len(group) == 1 for group in grouping.groups))

    def test_empty_sets_are_singletons_and_never_grouped(self):
        sets = [[], [], ["a", "b", "c"], ["a", "b", "c"]]
        grouping = group_near_duplicates(sets, jaccard_threshold=0.9)
        self.assertEqual(grouping.empty_records, [0, 1])
        self.assertEqual(grouping.groups, [[0], [1], [2, 3]])
        self.assertEqual(grouping.candidate_pairs, 1)

    def test_only_lsh_candidates_can_be_grouped(self):
        # Lowering lsh_threshold trades precision for recall but never adds
        # candidates: every verified pair must also be an LSH candidate, and
        # `candidate_pairs` is reported so the recall basis is visible.
        sets = [["a", "b", "c", "d"], ["a", "b", "c", "x"], ["q", "r", "s", "t"]]
        wide = group_near_duplicates(sets, jaccard_threshold=0.6, lsh_threshold=0.5)
        narrow = group_near_duplicates(sets, jaccard_threshold=0.6, lsh_threshold=0.9)
        self.assertGreaterEqual(wide.candidate_pairs, narrow.candidate_pairs)
        self.assertLessEqual(wide.candidate_pairs, 3)
        self.assertIn((0, 1), {pair[:2] for pair in wide.verified_pairs})

    def test_describe_mentions_pairs_and_empty_sets(self):
        grouping = group_near_duplicates([["a"], ["a"], []], jaccard_threshold=0.9)
        text = grouping.describe()
        self.assertIn("1 verified near-duplicate pairs", text)
        self.assertIn("1 empty-set records", text)


class TestSplitTrainValidation(unittest.TestCase):
    def test_returns_original_records_in_input_order(self):
        records = [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}]
        split = split_train_validation(records, [["a", "b"], ["a", "b"], ["c"], ["d"]], validation_ratio=0.5)
        self.assertIsInstance(split, TrainValidationSplit)
        self.assertIs(split.train[0], records[split.train_indices[0]])
        self.assertEqual(sorted(split.train_indices + split.validation_indices), [0, 1, 2, 3])
        self.assertEqual(split.train_indices, sorted(split.train_indices))
        self.assertEqual(split.validation_indices, sorted(split.validation_indices))

    def test_near_duplicates_never_cross_sides(self):
        records = ["a b c", "a b c", "x y", "z"]
        split = split_train_validation(records, labels=["p", "p", "n", "n"], validation_ratio=0.5)
        side = {index: "validation" if index in split.validation_indices else "train" for index in range(4)}
        self.assertEqual(side[0], side[1])
        self.assertEqual(split.grouping.groups, [[0, 1], [2], [3]])

    def test_total_deviation_has_priority_over_class_deviation(self):
        # One group of 3 rows, one of 1, one of 1. Ratio 0.4 of 5 rows -> 2.0.
        # Sending the 3-row group (deviation 1.0) is worse than 1+1 (deviation
        # 0.0); classes then decide between the two 1-row candidates.
        records = ["g1 a", "g1 a", "g1 a", "g2", "g3"]
        token_sets = [["a"], ["a"], ["a"], ["b"], ["c"]]
        labels = ["x", "x", "x", "y", "y"]
        split = split_train_validation(records, token_sets, labels, validation_ratio=0.4)
        self.assertEqual(split.actual_validation_rows, 2)
        self.assertAlmostEqual(split.total_deviation, 0.0)
        self.assertTrue(split.target_reachable)

    def test_tie_break_prefers_balanced_classes(self):
        records = list("abcd")
        token_sets = [["a"], ["b"], ["c"], ["d"]]
        labels = ["p", "p", "n", "n"]
        split = split_train_validation(records, token_sets, labels, validation_ratio=0.5)
        self.assertEqual(split.actual_validation_rows, 2)
        stats = {stat.label: stat for stat in split.class_stats}
        self.assertIsInstance(stats["p"], ClassSplitStats)
        self.assertEqual((stats["p"].validation, stats["n"].validation), (1, 1))
        self.assertAlmostEqual(split.class_deviation_sum, 0.0)

    def test_unreachable_target_is_reported_honestly(self):
        # A single group of 5 rows: validation is either 0 or 5, never 2.
        # 0 rows misses the 2-row target by 2, 5 rows by 3, so 0 is reported.
        records = ["a b c d"] * 5
        token_sets = [["a", "b", "c", "d"]] * 5
        split = split_train_validation(records, token_sets, ["p"] * 5, validation_ratio=0.4)
        self.assertFalse(split.target_reachable)
        self.assertEqual(len(split.grouping.groups), 1)
        self.assertEqual(split.actual_validation_rows, 0)
        self.assertAlmostEqual(split.total_deviation, 2.0)
        self.assertTrue(any("closest achievable total" in note for note in split.notes))
        self.assertTrue(any("kept intact" in note for note in split.notes))
        self.assertEqual(len(split.train) + len(split.validation), 5)

    def test_no_groups_are_split_and_no_rows_dropped(self):
        records = [f"r{i}" for i in range(9)]
        token_sets = [["a", "b"]] * 3 + [["c", "d"]] * 3 + [["e"]] * 3
        labels = ["p", "p", "p", "n", "n", "n", "m", "m", "m"]
        split = split_train_validation(records, token_sets, labels, validation_ratio=0.34)
        self.assertEqual(sorted(split.train_indices + split.validation_indices), list(range(9)))
        for group in split.grouping.groups:
            sides = {"validation" if index in split.validation_indices else "train" for index in group}
            self.assertEqual(len(sides), 1)

    def test_precomputed_grouping_is_reused(self):
        records = ["a b", "a b", "c", "d"]
        token_sets = [["a", "b"], ["a", "b"], ["c"], ["d"]]
        grouping = group_near_duplicates(token_sets, jaccard_threshold=0.9)
        first = split_train_validation(records, token_sets, ["p", "p", "n", "n"], 0.5, grouping=grouping)
        second = split_train_validation(records, token_sets, ["p", "p", "n", "n"], 0.5, grouping=grouping)
        self.assertIs(first.grouping, grouping)
        self.assertEqual(first.validation_indices, second.validation_indices)

    def test_mismatched_grouping_is_rejected(self):
        grouping = group_near_duplicates([["a"], ["b"]], jaccard_threshold=0.9)
        with self.assertRaises(ValueError):
            split_train_validation(["a b c"], [["a"], ["b"], ["c"]], None, 0.5, grouping=grouping)

    def test_labels_are_optional(self):
        split = split_train_validation(["a", "b", "c", "d"], [["a"], ["b"], ["c"], ["d"]], validation_ratio=0.5)
        self.assertIsNone(split.class_stats)
        self.assertAlmostEqual(split.class_deviation_sum, 0.0)
        self.assertEqual(split.actual_validation_rows, 2)

    def test_empty_input(self):
        split = split_train_validation([], [], None, 0.2)
        self.assertEqual((split.train, split.validation), ([], []))
        self.assertTrue(split.target_reachable)

    def test_default_tokenizer_requires_text(self):
        with self.assertRaises(TypeError):
            split_train_validation([1, 2], labels=None, validation_ratio=0.5)

    def test_ratio_bounds_are_validated(self):
        with self.assertRaises(ValueError):
            split_train_validation(["a"], [["a"]], None, 1.5)
        with self.assertRaises(ValueError):
            split_train_validation(["a"], [["a"]], None, 0.5, jaccard_threshold=2.0)

    def test_token_kinds_do_not_collide(self):
        grouping = group_near_duplicates([["1"], [1]], jaccard_threshold=0.9)
        self.assertEqual(grouping.groups, [[0], [1]])

    def test_report_is_readable(self):
        split = split_train_validation(["a", "a", "b"], [["a"], ["a"], ["b"]], ["x", "x", "y"], 0.5)
        text = split.report()
        self.assertIn("train rows: 1, validation rows: 2", text)
        self.assertIn("'x': target 1.000", text)
        self.assertIn("sum of class deviations:", text)
        self.assertIn("note:", text)


class TestValidationGroupSolver(unittest.TestCase):
    def test_matches_exhaustive_search(self):
        rng = random.Random(20240517)
        for _ in range(120):
            group_count = rng.randint(1, 9)
            sizes = [rng.randint(1, 5) for _ in range(group_count)]
            class_count = rng.randint(0, 3)
            counts = [[rng.randint(0, size) for _ in range(class_count)] for size in sizes]
            total = sum(sizes)
            ratio = rng.choice([0.0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 1.0, rng.random()])
            class_totals = [sum(row[class_index] for row in counts) for class_index in range(class_count)]
            total_target = ratio * total
            class_targets = [ratio * value for value in class_totals]

            groups, total_deviation, class_deviation_sum = _solve_validation_groups(
                sizes, counts, total_target, class_targets
            )
            expected_total, expected_class = _brute_force(sizes, counts, total_target, class_targets)

            rows = sum(sizes[index] for index in groups)
            achieved_class = sum(
                abs(sum(counts[index][class_index] for index in groups) - class_targets[class_index])
                for class_index in range(class_count)
            )
            self.assertAlmostEqual(abs(rows - total_target), expected_total, places=9)
            self.assertAlmostEqual(abs(rows - total_target), total_deviation, places=9)
            self.assertAlmostEqual(achieved_class, expected_class, places=6)
            self.assertAlmostEqual(class_deviation_sum, expected_class, places=6)

    def test_single_group(self):
        groups, total_deviation, class_deviation_sum = _solve_validation_groups([4], [[]], 2.0, [])
        self.assertIn(groups, ([], [0]))
        self.assertAlmostEqual(total_deviation, 2.0)
        self.assertAlmostEqual(class_deviation_sum, 0.0)

    def test_empty_groups(self):
        self.assertEqual(_solve_validation_groups([], [], 0.0, []), ([], 0.0, 0.0))

    def test_deterministic_output(self):
        args = ([3, 1, 1, 2], [[1, 0], [0, 1], [1, 0], [1, 1]], 2.0, [2.0, math.pi / 3])
        results = {tuple(_solve_validation_groups(*args)[0]) for _ in range(5)}
        self.assertEqual(len(results), 1)
