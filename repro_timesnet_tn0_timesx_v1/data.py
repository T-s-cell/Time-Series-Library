#!/usr/bin/env python3
"""Read-only window access over the exported LN frozen snapshot.

Every window (dense train, native val, native test) is a slice of the cached
float64 series in data_cache.npz - identical indexing to the LN run, so val/
test enumerations are window-for-window the same as F1. torch-free.
"""
import json
import os

import numpy as np

from common import CFG, CTX, PRED, SNAPSHOT, DOMAIN_NAMES, STD_EPS


class SnapshotError(RuntimeError):
    pass


class DenseData:
    def __init__(self, manifests_dir=None):
        d = manifests_dir or SNAPSHOT
        with open(os.path.join(d, "split_manifest.json")) as f:
            self.manifest = json.load(f)
        self.cache = np.load(os.path.join(d, "data_cache.npz"),
                             allow_pickle=False)
        self._keys = [str(k) for k in self.cache["var_keys"]]
        self._series_off = self.cache["series_off"]
        self._series_len = self.cache["series_len"]
        self._series = self.cache["series"]

        # load-time self checks (fail loud, never silent)
        tot = self.manifest["totals"]
        exp = CFG["budget"]["expected_totals"]
        for k in ("native_train", "dense", "val", "test", "excluded"):
            if tot.get(k) != exp[k]:
                raise SnapshotError(f"totals[{k}]={tot.get(k)} != {exp[k]}")
        if len(self._keys) != exp["n_vars"]:
            raise SnapshotError(f"{len(self._keys)} var keys != {exp['n_vars']}")
        if self._keys != list(self.manifest["variables"].keys()):
            raise SnapshotError("var_keys order != manifest variables order")
        if not np.isfinite(self._series).all():
            raise SnapshotError("non-finite values in cached series")
        self.fb_std = {}
        for vk in self._keys:
            fb = float(self.manifest["variables"][vk]["fallback_std"])
            if not np.isfinite(fb) or fb < CFG["data"]["std_eps"]:
                raise SnapshotError(f"{vk}: degenerate fallback_std {fb}")
            self.fb_std[vk] = fb

        self.domain_vars = {dm: sorted(vk for vk in self._keys
                                       if self.manifest["variables"][vk]
                                       ["domain"] == dm)
                            for dm in DOMAIN_NAMES}
        for dm, vs in self.domain_vars.items():
            row = next(r for r in CFG["budget"]["expected_domain_table"]
                       if r["domain"] == dm)
            if len(vs) != row["n_vars"]:
                raise SnapshotError(f"{dm}: {len(vs)} vars != {row['n_vars']}")

    def series(self, vk):
        i = self._keys.index(vk)
        o, n = int(self._series_off[i]), int(self._series_len[i])
        return self._series[o:o + n]

    def window(self, vk, s):
        ser = self.series(vk)
        return ser[s:s + CTX], ser[s + CTX:s + CTX + PRED]

    def native_ids(self, vk, split):
        return self.manifest["variables"][vk]["native"][split]

    def start_of(self, vk, sample_id):
        return self.manifest["variables"][vk]["sample_start_idx"][sample_id]

    def split_window_by_id(self, vk, sample_id):
        return self.window(vk, self.start_of(vk, sample_id))

    def dense_count(self, vk):
        return self.manifest["variables"][vk]["dense_count"]

    def domain_dense_total(self, domain):
        return sum(self.dense_count(vk) for vk in self.domain_vars[domain])

    def domain_native_total(self, domain, split):
        return sum(len(self.native_ids(vk, split))
                   for vk in self.domain_vars[domain])

    def split_rows(self, domain, split):
        """[(var_key, sample_id, x, y)] in stored (future-start) order."""
        rows = []
        for vk in self.domain_vars[domain]:
            for sid in self.native_ids(vk, split):
                s = self.start_of(vk, sid)
                x, y = self.window(vk, s)
                rows.append((vk, sid, x, y))
        return rows


def window_d(x, fb_std):
    """d_i = pstdev(raw 96-step input, ddof=0); fallback below std_eps."""
    s = float(np.std(x))
    return s if s >= STD_EPS else fb_std
