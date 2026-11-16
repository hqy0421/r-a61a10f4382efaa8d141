.. _train_validation_split:

Train/Validation Split Without Near-Duplicate Leakage
=====================================================

When labeled text is split into a training set and a validation set, a
near-duplicate of a training record that lands in validation makes the
validation score optimistic: the model is evaluated on text it has
effectively already seen. Random per-row splitting cannot prevent that,
because the two copies of a near-duplicate are assigned independently.

The :func:`datasketch.split_train_validation` workflow prevents it by
grouping first and splitting afterwards:

1. Each record's tokens are sketched with :class:`datasketch.MinHash` and
   indexed in a :class:`datasketch.MinHashLSH` index.
2. Each record queries that index. The returned candidate pairs are
   **verified with exact Jaccard similarity on the original token sets**
   -- the MinHash estimate is only used for recall.
3. Verified pairs are joined transitively. A record connected to another
   through a chain of verified pairs belongs to the same *group*, and a
   group is always assigned whole to one side.
4. Records whose token set is empty are never indexed and never linked,
   so empty sets do not automatically form a group; each is its own
   singleton group.
5. Groups are chosen for validation with these priorities, in order:

   a. minimize the absolute deviation between the validation row count and
      ``validation_ratio * total_rows``;
   b. then minimize the sum of per-class absolute deviations between
      each class's validation row count and
      ``validation_ratio * rows_of_that_class``.

Groups are never split and no record is ever dropped. The requested ratio
can therefore be unreachable; the result reports the actual counts, the
deviations, and a note that says the target was not reached and why,
rather than claiming success.

Candidate recall is the native, approximate MinHash LSH recall. Pairs the
index does not return are not grouped, so this workflow does **not**
guarantee finding every near-duplicate pair. Lower ``lsh_threshold``
(relative to ``jaccard_threshold``) or raise ``num_perm`` for better
recall; every recalled candidate is still verified exactly.

.. code:: python

        from datasketch import split_train_validation

        records = [
            "the cat sat on the mat",
            "the cat sat on the mat today",
            "a dog barked loudly",
            "a dog barked very loudly",
            "unrelated note",
        ]
        labels = ["pet", "pet", "pet", "pet", "misc"]

        split = split_train_validation(
            records,
            labels=labels,
            validation_ratio=0.4,
            jaccard_threshold=0.6,
        )
        print(split.report())
        split.validation  # original records, validation side
        split.train       # original records, train side
        split.grouping.groups  # grouping basis: transitive near-duplicate groups

Token sets can also be supplied directly -- useful when the records are
not plain text, or when tokens have already been extracted:

.. code:: python

        split = split_train_validation(
            records=[{"text": "...", "id": 1}, ...],
            token_sets=[doc_tokens_1, doc_tokens_2, ...],
            labels=[...],
            validation_ratio=0.2,
        )

If the same groups should be reused for several ratios, compute them once
with :func:`datasketch.group_near_duplicates` and pass the result back via
the ``grouping`` argument. A :class:`datasketch.NearDuplicateGrouping` also
exposes the verified pairs and the number of candidate pairs recalled, so
the grouping itself can be inspected.
