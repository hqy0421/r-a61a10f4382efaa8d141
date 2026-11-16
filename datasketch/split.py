"""Leakage-safe train/validation splitting with near-duplicate grouping.

Labeled records that are near-duplicates of each other must not end up on
different sides of a train/validation split: the model would then be evaluated
on text it has effectively already seen. This module groups records that are
transitively connected by near-duplicate links and assigns *whole groups* to
one side.

The pipeline uses the existing approximate index:

1. Every non-empty token set is sketched with :class:`datasketch.MinHash` and
   inserted into a :class:`datasketch.MinHashLSH` index.
2. Each record queries the index to recall candidate near-duplicate pairs.
3. Candidates are verified against the *original* token sets with an exact
   Jaccard threshold; only verified pairs create links.
4. Connected components of the verified link graph are the groups. A record
   whose token set is empty never links to anything, so empty sets do not
   automatically form a group: each empty-set record is its own group.
5. Whole groups are assigned to the validation side so that, in this order,

   a. the absolute deviation between the validation row count and
      ``validation_ratio * total_records`` is minimized, then
   b. the sum of absolute deviations between each class's validation row count
      and ``validation_ratio * class_records`` is minimized.

   Groups are never split and no record is ever dropped, so the requested
   ratio may be unreachable. The result reports the achieved deviations and
   explains any shortfall instead of pretending to hit the target.

Candidate recall is the native, approximate MinHash LSH recall, not an
all-pairs join: pairs that the index does not return are not grouped. Lowering
``lsh_threshold`` (or raising ``num_perm``) improves recall at the cost of more
candidate pairs, every one of which is still verified exactly.

Example:
    .. code-block:: python

        from datasketch.split import split_train_validation

        records = ["doc a", "doc b", "doc c", "doc d"]
        token_sets = [["a", "b", "c"], ["a", "b", "c"], ["x", "y"], ["z"]]
        labels = ["pos", "pos", "neg", "neg"]
        split = split_train_validation(records, token_sets, labels, 0.5)
        split.train  # the original objects, train side
        split.validation  # the original objects, validation side
        split.grouping.groups  # grouping basis: transitive near-duplicate groups
        split.class_stats  # per-class counts and deviations
        split.notes  # honest explanation of any missed target

"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from datasketch.lsh import MinHashLSH
from datasketch.minhash import MinHash

# Numerical slack used only to preserve the lexicographic priorities when the
# solver re-solves a pinned optimum. It is far below any difference that would
# be meaningful in a reported deviation.
_PIN_TOLERANCE = 1e-6


def _normalize_tokens(tokens: Iterable) -> frozenset[bytes]:
    """Turn a token iterable into the hashable byte set used by MinHash.

    Tokens are tagged by kind so that, for example, the string ``"1"`` and the
    integer ``1`` stay distinct instead of collapsing onto the same bytes. The
    ``str`` form of a token is used for kinds other than ``str`` and ``bytes``;
    the exact Jaccard check and the MinHash sketch both use this same set, so
    they always agree with each other.
    """
    normalized = set()
    for token in tokens:
        if isinstance(token, bytes):
            normalized.add(b"b:" + token)
        elif isinstance(token, str):
            normalized.add(b"s:" + token.encode("utf-8"))
        else:
            normalized.add(b"o:" + str(token).encode("utf-8"))
    return frozenset(normalized)


class _UnionFind:
    """Disjoint-set over record indices, used to build transitive groups."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


class NearDuplicateGrouping:
    """Transitive near-duplicate groups discovered for a set of records.

    Attributes:
        groups (list[list[int]]): Record indices grouped by transitive
            near-duplicate links, ordered by their smallest member index.
            Records with an empty token set each form a singleton group.
        group_of (list[int]): For each record index, the position of its group
            in :attr:`groups`.
        verified_pairs (list[tuple[int, int, float]]): Candidate pairs that
            passed the exact Jaccard check, as
            ``(smaller_index, larger_index, jaccard)`` sorted by indices.
        candidate_pairs (int): Number of candidate pairs recalled by
            MinHashLSH before exact verification.
        rejected_pairs (int): Candidate pairs rejected by exact verification.
        empty_records (list[int]): Indices of records with an empty token set.
        lsh_threshold (float): Jaccard threshold the LSH recall was built for.
        jaccard_threshold (float): Exact Jaccard threshold used to verify.
        num_perm (int): Number of MinHash permutations.

    """

    def __init__(
        self,
        groups: list[list[int]],
        group_of: list[int],
        verified_pairs: list[tuple[int, int, float]],
        candidate_pairs: int,
        rejected_pairs: int,
        empty_records: list[int],
        lsh_threshold: float,
        jaccard_threshold: float,
        num_perm: int,
    ) -> None:
        self.groups = groups
        self.group_of = group_of
        self.verified_pairs = verified_pairs
        self.candidate_pairs = candidate_pairs
        self.rejected_pairs = rejected_pairs
        self.empty_records = empty_records
        self.lsh_threshold = lsh_threshold
        self.jaccard_threshold = jaccard_threshold
        self.num_perm = num_perm

    @property
    def group_sizes(self) -> list[int]:
        """Number of records in each group, in :attr:`groups` order."""
        return [len(group) for group in self.groups]

    @property
    def num_non_singleton_groups(self) -> int:
        """Number of groups that link more than one record."""
        return sum(1 for group in self.groups if len(group) > 1)

    def describe(self) -> str:
        """Return a short human-readable summary of the grouping."""
        return (
            f"{len(self.groups)} groups from {sum(self.group_sizes)} records: "
            f"{self.num_non_singleton_groups} non-singleton, "
            f"{len(self.empty_records)} empty-set records kept as singletons; "
            f"{len(self.verified_pairs)} verified near-duplicate pairs out of "
            f"{self.candidate_pairs} LSH candidates (Jaccard >= {self.jaccard_threshold:g})"
        )


def _default_tokenizer(record) -> Iterable:
    """Split a text record on whitespace; the default when no tokens are given."""
    if isinstance(record, (str, bytes)):
        return record.split()
    raise TypeError(
        "The default tokenizer only handles str or bytes records; pass token_sets= or tokenizer= "
        "for records of type %s" % type(record).__name__
    )


def _validate_thresholds(jaccard_threshold: float, lsh_threshold: Optional[float]) -> None:
    if not 0.0 <= jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be in [0.0, 1.0]")
    if lsh_threshold is not None and not 0.0 <= lsh_threshold <= 1.0:
        raise ValueError("lsh_threshold must be in [0.0, 1.0]")


def group_near_duplicates(
    token_sets: Sequence[Iterable],
    *,
    jaccard_threshold: float = 0.8,
    lsh_threshold: Optional[float] = None,
    num_perm: int = 128,
    weights: tuple[float, float] = (0.5, 0.5),
) -> NearDuplicateGrouping:
    """Group records whose token sets are transitively near-duplicate.

    Candidate pairs come from a :class:`datasketch.MinHashLSH` index built over
    MinHash sketches of the token sets; each candidate is verified against the
    original token sets with exact Jaccard similarity. Verified pairs are joined
    transitively, so the groups are the connected components of the verified
    pair graph.

    Records whose token set is empty are never inserted into the index and never
    linked to any other record, including other empty ones: each such record is
    its own singleton group.

    Args:
        token_sets (Sequence[Iterable]): One iterable of tokens per record.
        jaccard_threshold (float): Exact Jaccard threshold a candidate pair must
            meet to be linked, ``0.0 <= threshold <= 1.0``.
        lsh_threshold (Optional[float]): Jaccard threshold the MinHashLSH index
            is optimized for, used only to recall candidates. Defaults to
            `jaccard_threshold`; pass a lower value to improve candidate recall.
            Recall remains approximate: pairs absent from the index are not
            grouped.
        num_perm (int): Number of MinHash permutations.
        weights (Tuple[float, float]): False-positive/false-negative weights for
            the LSH parameter optimization, as in :class:`datasketch.MinHashLSH`.

    Returns:
        NearDuplicateGrouping: The groups and the details of how they were formed.

    """
    _validate_thresholds(jaccard_threshold, lsh_threshold)
    if num_perm < 2:
        raise ValueError("num_perm must be at least 2")
    if lsh_threshold is None:
        lsh_threshold = jaccard_threshold

    normalized = [_normalize_tokens(tokens) for tokens in token_sets]
    record_count = len(normalized)

    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm, weights=weights)
    minhashes: dict[int, MinHash] = {}
    for index, token_set in enumerate(normalized):
        if not token_set:
            continue
        sketch = MinHash(num_perm=num_perm)
        sketch.update_batch(token_set)
        minhashes[index] = sketch
        lsh.insert(index, sketch)

    candidate_pairs = set()
    for index, sketch in minhashes.items():
        for other in lsh.query(sketch):
            if other != index:
                candidate_pairs.add((index, other) if index < other else (other, index))

    verified_pairs = []
    rejected_pairs = 0
    for left, right in sorted(candidate_pairs):
        left_set, right_set = normalized[left], normalized[right]
        union_size = len(left_set | right_set)
        if union_size == 0:  # unreachable: empty sets are never indexed
            continue
        similarity = len(left_set & right_set) / union_size
        if similarity >= jaccard_threshold:
            verified_pairs.append((left, right, similarity))
        else:
            rejected_pairs += 1

    union_find = _UnionFind(record_count)
    for left, right, _similarity in verified_pairs:
        union_find.union(left, right)

    members: dict[int, list[int]] = {}
    for index in range(record_count):
        members.setdefault(union_find.find(index), []).append(index)
    groups = sorted(members.values(), key=lambda group: group[0])
    group_of = [0] * record_count
    for position, group in enumerate(groups):
        for index in group:
            group_of[index] = position

    empty_records = [index for index, token_set in enumerate(normalized) if not token_set]
    return NearDuplicateGrouping(
        groups=groups,
        group_of=group_of,
        verified_pairs=verified_pairs,
        candidate_pairs=len(candidate_pairs),
        rejected_pairs=rejected_pairs,
        empty_records=empty_records,
        lsh_threshold=lsh_threshold,
        jaccard_threshold=jaccard_threshold,
        num_perm=num_perm,
    )


@dataclass(frozen=True)
class ClassSplitStats:
    """Validation/train counts and target deviation for one label.

    Attributes:
        label (Hashable): The class label.
        total (int): Number of records with this label.
        train (int): Number of train records with this label.
        validation (int): Number of validation records with this label.
        target_validation (float): ``validation_ratio * total``, the requested
            validation count for this class.
        target_deviation (float): ``abs(validation - target_validation)``.

    """

    label: Hashable
    total: int
    train: int
    validation: int
    target_validation: float
    target_deviation: float


@dataclass(frozen=True)
class TrainValidationSplit:
    """Result of :func:`split_train_validation`.

    The two record lists contain the *original* record objects, in input order,
    so a caller can write them out unchanged. Nothing is dropped or duplicated:
    the two index lists partition ``range(len(records))``.

    Attributes:
        train (list): Original records assigned to train, in input order.
        validation (list): Original records assigned to validation, in input
            order.
        train_indices (list[int]): Indices of the train records.
        validation_indices (list[int]): Indices of the validation records.
        grouping (NearDuplicateGrouping): The grouping basis: transitive
            near-duplicate groups and the pairs that produced them.
        validation_groups (list[int]): Positions in ``grouping.groups`` of the
            groups assigned to validation.
        train_groups (list[int]): Positions in ``grouping.groups`` of the groups
            assigned to train.
        validation_ratio (float): The requested validation fraction.
        target_validation_rows (float): ``validation_ratio * len(records)``.
        actual_validation_rows (int): Number of validation records.
        total_deviation (float): ``abs(actual_validation_rows -
            target_validation_rows)``.
        class_stats (Optional[list[ClassSplitStats]]): Per-class counts and
            deviations, in order of first appearance of the label; ``None`` when
            no labels were given.
        class_deviation_sum (float): Sum of the per-class target deviations.
        target_reachable (bool): Whether the validation total meets the target
            exactly. When ``False``, no group-respecting split can, and
            :attr:`notes` says so.
        notes (list[str]): Honest remarks about unreachable targets and the
            approximation used for grouping.

    """

    train: list
    validation: list
    train_indices: list[int]
    validation_indices: list[int]
    grouping: NearDuplicateGrouping
    validation_groups: list[int]
    train_groups: list[int]
    validation_ratio: float
    target_validation_rows: float
    actual_validation_rows: int
    total_deviation: float
    class_stats: Optional[list[ClassSplitStats]]
    class_deviation_sum: float
    target_reachable: bool
    notes: list[str]

    def report(self) -> str:
        """Return a readable summary of the split and its deviations."""
        lines = [
            "train rows: %d, validation rows: %d, total: %d"
            % (len(self.train), len(self.validation), len(self.train) + len(self.validation)),
            "target validation rows: %.3f (ratio %.4f), actual: %d, deviation: %.3f"
            % (
                self.target_validation_rows,
                self.validation_ratio,
                self.actual_validation_rows,
                self.total_deviation,
            ),
            "grouping: " + self.grouping.describe(),
        ]
        if self.class_stats is not None:
            lines.append("class deviations (target, actual, |deviation|):")
            for stat in self.class_stats:
                lines.append(
                    "  %r: target %.3f, train %d, validation %d, deviation %.3f"
                    % (stat.label, stat.target_validation, stat.train, stat.validation, stat.target_deviation)
                )
            lines.append("sum of class deviations: %.3f" % self.class_deviation_sum)
        for note in self.notes:
            lines.append("note: " + note)
        return "\n".join(lines)


def _solve_validation_groups(
    group_rows: Sequence[int],
    group_class_counts: Sequence[Sequence[int]],
    total_target: float,
    class_targets: Sequence[float],
) -> tuple[list[int], float, float]:
    """Choose which groups go to validation, following the required priorities.

    Groups with the same size and class composition are interchangeable for the
    objective, so they are collapsed into one integer variable counting how many
    of them go to validation. This keeps the program small even when most
    records are singleton groups after grouping.

    Stage 1 minimizes ``abs(validation_rows - total_target)``. Stage 2 minimizes
    the sum of per-class absolute deviations over all assignments that are
    optimal in stage 1. A final stage only breaks remaining ties (preferring the
    groups that appear earlier in the input), so the reported assignment is
    reproducible.

    Returns:
        tuple: ``(validation_group_positions, total_deviation,
        class_deviation_sum)``.

    """
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:  # pragma: no cover - scipy < 1.9
        raise ImportError("split_train_validation requires scipy.optimize.milp (scipy >= 1.9)") from exc

    group_count = len(group_rows)
    class_count = len(class_targets)
    if group_count == 0:
        return [], abs(total_target), 0.0
    sizes = np.asarray(group_rows, dtype=float)
    counts = np.asarray(group_class_counts, dtype=float).reshape(group_count, class_count)
    total_rows = int(sizes.sum())

    # Collapse interchangeable groups into types: same row count and same
    # per-class counts. Members of a type are kept in index order so the final
    # assignment always fills a type's earliest groups first.
    type_members: dict[tuple, list[int]] = {}
    for index in range(group_count):
        signature = (int(sizes[index]), *(int(value) for value in counts[index]))
        type_members.setdefault(signature, []).append(index)
    member_lists = list(type_members.values())
    type_signatures = list(type_members)
    type_sizes = np.asarray([signature[0] for signature in type_signatures], dtype=float)
    type_counts = np.asarray([signature[1:] for signature in type_signatures], dtype=float)
    type_upper = np.asarray([len(members) for members in member_lists], dtype=float)
    type_count = len(member_lists)
    # Cheap, deterministic secondary criterion: prefer types whose first member
    # appears earlier. It never overrides either objective (see stage 3).
    type_positions = np.asarray([members[0] for members in member_lists], dtype=float)

    # Stage 1: closest reachable validation row count.
    stage1_cost = np.zeros(type_count + 1)
    stage1_cost[-1] = 1.0
    stage1_lower = np.zeros(type_count + 1)
    stage1_upper = np.concatenate([type_upper, [np.inf]])
    # V - d1 <= T and T - V <= d1, with d1 >= 0.
    stage1_matrix = csr_matrix(
        np.vstack(
            [
                np.concatenate([type_sizes, [-1.0]]),
                np.concatenate([-type_sizes, [-1.0]]),
            ]
        )
    )
    stage1 = milp(
        stage1_cost,
        integrality=np.concatenate([np.ones(type_count), [0.0]]),
        bounds=Bounds(stage1_lower, stage1_upper),
        constraints=[
            LinearConstraint(
                stage1_matrix,
                np.array([-np.inf, -np.inf]),
                np.array([total_target, -total_target]),
            )
        ],
    )
    if not stage1.success:
        raise RuntimeError("could not optimize the validation row target: %s" % stage1.message)

    # Totals are integers, so the optimal ones lie in this (tight) interval.
    lowest_rows = max(0, int(np.ceil(total_target - float(stage1.x[-1]) - 1e-6)))
    highest_rows = min(total_rows, int(np.floor(total_target + float(stage1.x[-1]) + 1e-6)))
    if lowest_rows > highest_rows:  # numerical safety net
        nearest = min(max(round(total_target), 0), total_rows)
        lowest_rows = highest_rows = nearest

    # Stage 2: among those, the smallest sum of per-class deviations.
    stage2_vars = type_count + 2 * class_count
    stage2_cost = np.zeros(stage2_vars)
    stage2_cost[type_count:] = 1.0
    stage2_lower = np.zeros(stage2_vars)
    stage2_upper = np.concatenate([type_upper, np.full(2 * class_count, np.inf)])
    rows: list[list[float]] = []
    row_lower: list[float] = []
    row_upper: list[float] = []
    total_row = np.zeros(stage2_vars)
    total_row[:type_count] = type_sizes
    rows.append(total_row.tolist())
    row_lower.append(lowest_rows)
    row_upper.append(highest_rows)
    rows.append((-total_row).tolist())
    row_lower.append(-highest_rows)
    row_upper.append(-lowest_rows)
    for class_index in range(class_count):
        row = np.zeros(stage2_vars)
        row[:type_count] = type_counts[:, class_index]
        # V_c - p + q = T_c, with p, q >= 0 and the objective summing p + q.
        row[type_count + class_index] = -1.0
        row[type_count + class_count + class_index] = 1.0
        rows.append(row.tolist())
        row_lower.append(float(class_targets[class_index]))
        row_upper.append(float(class_targets[class_index]))
    stage2 = milp(
        stage2_cost,
        integrality=np.concatenate([np.ones(type_count), np.zeros(2 * class_count)]),
        bounds=Bounds(stage2_lower, stage2_upper),
        constraints=[LinearConstraint(csr_matrix(np.asarray(rows)), np.asarray(row_lower), np.asarray(row_upper))],
    )
    if not stage2.success:
        raise RuntimeError("could not optimize the per-class validation counts: %s" % stage2.message)

    def select_and_measure(type_values) -> tuple[list[int], float, float]:
        """Expand type counts into group indices and recompute the achievements.

        Deviations are computed from the actual assignment rather than read
        from the solver objective, whose feasibility tolerance is looser than
        what is reported.
        """
        selected = []
        for type_index in range(type_count):
            taken = round(float(type_values[type_index]))
            if taken > 0:
                selected.extend(member_lists[type_index][:taken])
        if selected:
            validation_rows = sum(int(sizes[index]) for index in selected)
            class_deviations = sum(
                abs(sum(int(counts[index, class_index]) for index in selected) - class_targets[class_index])
                for class_index in range(class_count)
            )
        else:
            validation_rows = 0
            class_deviations = sum(abs(target) for target in class_targets)
        return selected, abs(validation_rows - total_target), class_deviations

    best_groups, best_total_deviation, best_class_deviation = select_and_measure(stage2.x)

    # Stage 3: deterministic tie-break. Re-solve with the class-deviation
    # objective pinned to its optimum, minimizing the sum of first-member
    # positions of the chosen types. This only picks *which* equally optimal
    # assignment is reported. If the re-solve fails, stage 2's solution stands.
    tie_rows = list(rows)
    tie_lower = list(row_lower)
    tie_upper = list(row_upper)
    tie_row = np.zeros(stage2_vars)
    tie_row[type_count:] = 1.0
    tie_rows.append(tie_row.tolist())
    tie_lower.append(-np.inf)
    tie_upper.append(best_class_deviation + _PIN_TOLERANCE + 1e-9 * best_class_deviation)
    tie_cost = np.zeros(stage2_vars)
    tie_cost[:type_count] = type_positions
    tie_solution = milp(
        tie_cost,
        integrality=np.concatenate([np.ones(type_count), np.zeros(2 * class_count)]),
        bounds=Bounds(stage2_lower, stage2_upper),
        constraints=[LinearConstraint(csr_matrix(np.asarray(tie_rows)), np.asarray(tie_lower), np.asarray(tie_upper))],
    )
    if tie_solution.success:
        candidate_groups, candidate_total, candidate_class = select_and_measure(tie_solution.x)
        # Both solutions are optimal; prefer the tie-broken one, keeping the
        # stage-2 solution only if the re-solve came out worse (the pin has a
        # tolerance and the solver's own feasibility tolerance is looser).
        if (
            candidate_total <= best_total_deviation + _PIN_TOLERANCE
            and candidate_class <= best_class_deviation + _PIN_TOLERANCE
        ):
            best_groups, best_total_deviation, best_class_deviation = (
                candidate_groups,
                candidate_total,
                candidate_class,
            )

    return sorted(best_groups), best_total_deviation, best_class_deviation


def split_train_validation(
    records: Sequence,
    token_sets: Optional[Sequence[Iterable]] = None,
    labels: Optional[Sequence[Hashable]] = None,
    validation_ratio: float = 0.2,
    *,
    tokenizer=None,
    grouping: Optional[NearDuplicateGrouping] = None,
    jaccard_threshold: float = 0.8,
    lsh_threshold: Optional[float] = None,
    num_perm: int = 128,
    weights: tuple[float, float] = (0.5, 0.5),
) -> TrainValidationSplit:
    """Split labeled records into train and validation without cross-side near-duplicates.

    The split is leakage-safe by construction: near-duplicates are found with
    :class:`datasketch.MinHashLSH` over MinHash sketches, verified with exact
    Jaccard similarity on the original token sets, joined transitively into
    groups, and each group is assigned whole to one side.

    Selection priority, applied to whole groups only:

    1. minimize ``abs(validation_rows - validation_ratio * len(records))``;
    2. then minimize the sum over classes of
       ``abs(validation_rows_of_class - validation_ratio * rows_of_class)``.

    Groups are never split and records are never dropped, so the targets may be
    unreachable. Achieved numbers and deviations are always reported, and
    :attr:`TrainValidationSplit.notes` states when and why the target was missed
    instead of claiming success.

    Args:
        records (Sequence): The original records, returned as-is in the two
            result lists.
        token_sets (Optional[Sequence[Iterable]]): One token iterable per record,
            used for grouping. If omitted, `tokenizer(record)` is used; the
            default tokenizer splits text on whitespace and rejects other types.
        labels (Optional[Sequence[Hashable]]): One class label per record; when
            given, the second priority applies and per-class statistics are
            reported.
        validation_ratio (float): Requested fraction of records in validation,
            ``0.0 <= ratio <= 1.0``.
        tokenizer (Optional[Callable]): Maps a record to an iterable of tokens;
            ignored when `token_sets` is given.
        grouping (Optional[NearDuplicateGrouping]): Precomputed grouping, e.g.
            from :func:`group_near_duplicates` when the same groups are reused
            for several ratios. Must cover the same records.
        jaccard_threshold (float): Exact Jaccard threshold for verifying a
            candidate pair as a near-duplicate.
        lsh_threshold (Optional[float]): Jaccard threshold the MinHashLSH recall
            is tuned for; defaults to `jaccard_threshold`. Lower values recall
            more candidate pairs, all of which are still verified exactly.
            Recall is approximate and not guaranteed complete.
        num_perm (int): Number of MinHash permutations.
        weights (Tuple[float, float]): LSH false-positive/false-negative weights.

    Returns:
        TrainValidationSplit: Both record lists, the grouping basis, per-class
        counts and deviations, and notes.

    Raises:
        ValueError: If the inputs have inconsistent lengths, the ratio or a
            threshold is out of range, or a supplied grouping does not match.

    Example:
        .. code-block:: python

            from datasketch.split import split_train_validation

            split = split_train_validation(
                records=["a b c", "a b c", "x y", "z", ""],
                labels=["pos", "pos", "neg", "neg", "neg"],
                validation_ratio=0.4,
            )
            print(split.report())

    """
    if not 0.0 <= validation_ratio <= 1.0:
        raise ValueError("validation_ratio must be in [0.0, 1.0]")
    _validate_thresholds(jaccard_threshold, lsh_threshold)
    record_count = len(records)
    notes: list[str] = []

    if token_sets is None:
        tokenizer = tokenizer or _default_tokenizer
        token_sets = [tokenizer(record) for record in records]
    elif len(token_sets) != record_count:
        raise ValueError("token_sets has %d entries but records has %d" % (len(token_sets), record_count))
    if labels is not None and len(labels) != record_count:
        raise ValueError("labels has %d entries but records has %d" % (len(labels), record_count))

    if grouping is None:
        grouping = group_near_duplicates(
            token_sets,
            jaccard_threshold=jaccard_threshold,
            lsh_threshold=lsh_threshold,
            num_perm=num_perm,
            weights=weights,
        )
    else:
        covered = sorted(index for group in grouping.groups for index in group)
        if covered != list(range(record_count)):
            raise ValueError("grouping must cover each of the %d records exactly once" % record_count)

    group_count = len(grouping.groups)
    group_of = [0] * record_count
    for position, group in enumerate(grouping.groups):
        for index in group:
            group_of[index] = position
    group_rows = [len(group) for group in grouping.groups]
    total_target = validation_ratio * record_count

    label_list = list(labels) if labels is not None else None
    if label_list is None:
        class_targets: list[float] = []
        group_class_counts: list[list[int]] = [[] for _ in range(group_count)]
    else:
        label_order = list(dict.fromkeys(label_list))
        label_index = {label: position for position, label in enumerate(label_order)}
        class_totals = Counter(label_list)
        class_targets = [validation_ratio * class_totals[label] for label in label_order]
        group_class_counts = [[0] * len(label_order) for _ in range(group_count)]
        for index, label in enumerate(label_list):
            group_class_counts[group_of[index]][label_index[label]] += 1

    validation_groups, total_deviation, class_deviation_sum = _solve_validation_groups(
        group_rows, group_class_counts, total_target, class_targets
    )

    validation_set = set(validation_groups)
    validation_indices = []
    train_indices = []
    for index in range(record_count):
        (validation_indices if group_of[index] in validation_set else train_indices).append(index)

    actual_rows = len(validation_indices)
    target_reachable = total_deviation <= 1e-9

    class_stats = None
    if label_list is not None:
        class_stats = []
        for label in label_order:
            total = class_totals[label]
            validation_count = sum(1 for index in validation_indices if label_list[index] == label)
            class_stats.append(
                ClassSplitStats(
                    label=label,
                    total=total,
                    train=total - validation_count,
                    validation=validation_count,
                    target_validation=validation_ratio * total,
                    target_deviation=abs(validation_count - validation_ratio * total),
                )
            )

    if grouping.num_non_singleton_groups:
        notes.append(
            "%d near-duplicate groups (%d records) were kept intact, so the requested ratio may be "
            "unreachable; recorded deviation is the best achievable with whole groups."
            % (grouping.num_non_singleton_groups, sum(size for size in group_rows if size > 1))
        )
    if not target_reachable:
        nearest_integer = round(total_target)
        achievable = (
            "no assignment of whole groups yields exactly %d validation rows" % nearest_integer
            if abs(total_target - nearest_integer) <= _PIN_TOLERANCE
            else "the target %.3f validation rows is not a whole number" % total_target
        )
        notes.append(
            "%s; the closest achievable total is %d rows (deviation %.3f)." % (achievable, actual_rows, total_deviation)
        )
    elif class_stats is not None and class_deviation_sum > 1e-9:
        notes.append(
            "the overall row target is met exactly, but per-class deviations remain (sum %.3f) "
            "because grouping constrains how rows can be distributed across classes." % class_deviation_sum
        )
    notes.append(
        "candidate recall is the native, approximate MinHashLSH recall; near-duplicate pairs the "
        "index does not return are not grouped, so completeness of near-duplicate detection is not "
        "guaranteed."
    )

    return TrainValidationSplit(
        train=[records[index] for index in train_indices],
        validation=[records[index] for index in validation_indices],
        train_indices=train_indices,
        validation_indices=validation_indices,
        grouping=grouping,
        validation_groups=list(validation_groups),
        train_groups=[index for index in range(group_count) if index not in validation_set],
        validation_ratio=validation_ratio,
        target_validation_rows=total_target,
        actual_validation_rows=actual_rows,
        total_deviation=total_deviation,
        class_stats=class_stats,
        class_deviation_sum=class_deviation_sum,
        target_reachable=target_reachable,
        notes=notes,
    )
