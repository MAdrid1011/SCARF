"""
Cache Table

Semantic-indexed cache with Hamming distance lookup.
"""
import torch
from typing import Optional, Tuple, List

from .types import FSDRConfig, CacheEntry
from .lsh_hasher import hamming_distance


class CacheTable:
    """
    Semantic-indexed cache table for FSDR.
    
    Features:
        - Parallel Hamming distance calculation for all entries
        - Best-match selection based on minimum Hamming distance
        - LRU replacement with confidence weighting
        - Entry update with exponential moving average
    
    Hardware Mapping:
        - 128 entries × 70 bits = 1.1KB SRAM
        - 128 parallel Hamming calculators (XOR + popcount)
        - Min-selector tree for best match
        - LRU counters for replacement policy
        - Total: ~300 LUTs, 1 cycle lookup latency
    
    Example:
        table = CacheTable(config)
        entry, hamming = table.lookup(signature)
        if entry is None:
            table.insert(new_entry)
    """
    
    def __init__(self, config: FSDRConfig):
        """
        Initialize cache table.
        
        Args:
            config: FSDR configuration
        """
        self.config = config
        self.size = config.cache_size
        self.hamming_threshold = config.hamming_threshold
        
        # Storage: list of CacheEntry (or None for empty slots)
        self.entries: List[Optional[CacheEntry]] = [None] * self.size
        
        # LRU timestamp counter
        self.time = 0
    
    def lookup(self, signature: int) -> Tuple[Optional[CacheEntry], int]:
        """
        Find best matching entry by Hamming distance.
        
        Args:
            signature: Query signature (K-bit integer)
        
        Returns:
            (entry, hamming_distance) if hit with distance <= threshold
            (None, -1) if miss
        
        Hardware:
            - 128 parallel XOR + popcount operations
            - Min-selector tree (7 levels for 128 entries)
            - Threshold comparator
            - 1 cycle latency
        """
        min_hamming = self.hamming_threshold + 1
        best_entry = None
        best_idx = -1
        
        for i, entry in enumerate(self.entries):
            if entry is None or not entry.valid:
                continue
            
            # Compute Hamming distance
            hamming = hamming_distance(signature, entry.signature)
            
            if hamming < min_hamming:
                min_hamming = hamming
                best_entry = entry
                best_idx = i
        
        if min_hamming <= self.hamming_threshold and best_entry is not None:
            # Update LRU timestamp
            best_entry.last_access = self.time
            self.time += 1
            return best_entry, min_hamming
        
        return None, -1
    
    def insert(self, entry: CacheEntry) -> int:
        """
        Insert new entry into cache.
        
        Uses LRU with confidence weighting for replacement.
        
        Args:
            entry: New cache entry
        
        Returns:
            Index where entry was inserted
        
        Hardware:
            - Find empty slot (parallel valid bit check)
            - Or find LRU among low-confidence entries
            - Or find global LRU
            - 1-2 cycles latency
        """
        # Set access time
        entry.last_access = self.time
        self.time += 1
        
        # Strategy 1: Find empty slot
        for i, e in enumerate(self.entries):
            if e is None or not e.valid:
                self.entries[i] = entry
                return i
        
        # Strategy 2: Find LRU among low-confidence entries
        low_conf_candidates = [
            (i, e.last_access) for i, e in enumerate(self.entries)
            if e is not None and e.peak_prob < 0.6
        ]
        
        if low_conf_candidates:
            # Select oldest among low confidence
            idx = min(low_conf_candidates, key=lambda x: x[1])[0]
            self.entries[idx] = entry
            return idx
        
        # Strategy 3: Pure LRU
        min_access = float('inf')
        lru_idx = 0
        for i, e in enumerate(self.entries):
            if e is not None and e.last_access < min_access:
                min_access = e.last_access
                lru_idx = i
        
        self.entries[lru_idx] = entry
        return lru_idx
    
    def update(
        self,
        entry: CacheEntry,
        new_depth: float,
        *,
        new_signature: int,
        new_position: Tuple[int, int],
    ):
        """
        Write the current feature signature and final depth into a hit entry.
        
        Args:
            entry: Entry to update (must be in cache)
            new_depth: New depth estimate
            new_signature: Current pixel's LSH signature
            new_position: Current pixel's raster position
        
        Hardware:
            - 1 multiply, 2 adds
            - In-place SRAM update
        """
        entry.signature = int(new_signature)
        entry.position = new_position
        entry.best_depth = float(new_depth)
        entry.last_access = self.time
        self.time += 1
    
    def invalidate(self, signature: int):
        """
        Invalidate entry with given signature.
        
        Args:
            signature: Signature of entry to invalidate
        """
        for entry in self.entries:
            if entry is not None and entry.signature == signature:
                entry.valid = False
                return
    
    def clear(self):
        """Clear all entries in cache."""
        self.entries = [None] * self.size
        self.time = 0
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        valid_count = sum(1 for e in self.entries if e is not None and e.valid)
        
        confidence_sum = sum(
            e.peak_prob for e in self.entries if e is not None and e.valid
        )
        avg_confidence = confidence_sum / valid_count if valid_count > 0 else 0
        
        return {
            'valid_entries': valid_count,
            'total_slots': self.size,
            'occupancy': valid_count / self.size,
            'avg_confidence': avg_confidence,
        }
    
    def __len__(self) -> int:
        """Return number of valid entries."""
        return sum(1 for e in self.entries if e is not None and e.valid)
    
    def __getitem__(self, idx: int) -> Optional[CacheEntry]:
        """Get entry at index."""
        return self.entries[idx]
