import importlib.metadata
from typing import Final

try:
    _version = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    _version = "0.0.0"  # Fallback for development mode
__version__: Final[str] = _version

from datasketch.aio import AsyncMinHashLSH  # Instantiation requires motor/redis.asyncio; import itself is always safe.
from datasketch.b_bit_minhash import bBitMinHash
from datasketch.hashfunc import sha1_hash32, sha1_hash64
from datasketch.hnsw import HNSW
from datasketch.hyperloglog import HyperLogLog, HyperLogLogPlusPlus
from datasketch.lean_minhash import LeanMinHash
from datasketch.lsh import MinHashLSH
from datasketch.lsh_bloom import MinHashLSHBloom
from datasketch.lshensemble import MinHashLSHEnsemble
from datasketch.lshforest import MinHashLSHForest
from datasketch.minhash import MinHash
from datasketch.split import (
    ClassSplitStats,
    NearDuplicateGrouping,
    TrainValidationSplit,
    group_near_duplicates,
    split_train_validation,
)
from datasketch.weighted_minhash import WeightedMinHash, WeightedMinHashGenerator

# Alias
WeightedMinHashLSH = MinHashLSH
WeightedMinHashLSHForest = MinHashLSHForest

__all__ = [
    "HNSW",
    "AsyncMinHashLSH",
    "ClassSplitStats",
    "HyperLogLog",
    "HyperLogLogPlusPlus",
    "LeanMinHash",
    "MinHash",
    "MinHashLSH",
    "MinHashLSHBloom",
    "MinHashLSHEnsemble",
    "MinHashLSHForest",
    "NearDuplicateGrouping",
    "TrainValidationSplit",
    "WeightedMinHash",
    "WeightedMinHashGenerator",
    "WeightedMinHashLSH",
    "WeightedMinHashLSHForest",
    "bBitMinHash",
    "group_near_duplicates",
    "sha1_hash32",
    "sha1_hash64",
    "split_train_validation",
]
