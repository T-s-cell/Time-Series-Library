# Vendored TimesNet — provenance

- Upstream project: thuml/Time-Series-Library (official TimesNet forecasting
  implementation lives in `models/TimesNet.py` + `layers/{Embed,Conv_Blocks}.py`).
- Baseline code version pinned for TN0: commit
  `74e58ebff7826f43ae0c6b979970c95627456358` (GitHub `T-s-cell/Time-Series-Library`,
  main). The three model source files at that commit are **byte-identical** to
  the same paths at `1d086314ccfa921ffb2388e5448851f635f09d52` (the commit the
  theta working tree is pinned at); equality was verified on 2026-10-03 by
  exporting both revisions independently and comparing sha256
  (`git show <rev>:<path>` on both sides). The TSLib repo carries these files
  unmodified from the thumn lineage used by the accepted T0/P0 campaigns.
- `upstream/` holds the pristine copies (used ONLY by `model_check.py`'s
  subprocess equivalence proof). `adapted/` holds the runtime copies.
- Adaptation is import-line package isolation ONLY: `models/` -> `tn_models/`,
  `layers/` -> `tn_layers/`, so the vendored packages can never collide with
  the TSLib repo's own top-level `models`/`layers` packages in one process.
  Exactly 2 lines change (both in `tn_models/TimesNet.py`); see
  `import_diff.patch`. State-dict keys contain no package names, so weights
  transfer 1:1 — proven bitwise by `model_check.py` (same state_dict + same
  inputs -> byte-identical outputs on both sides).

## Files (sha256)

| file | sha256 |
| --- | --- |
| upstream/LICENSE | 8a6caa178ea3f33ebff5d7bb5558628cf5b423305dc30d3630f86564c7db94a2 |
| upstream/models/TimesNet.py | f64c4bed1fd7347090044a0163bd4209c9f9ec1c1b19ceff47842df36b64bba7 |
| upstream/models/__init__.py | d97d758fc2ef4da5eb24a20e940538bb5b8f6ed81d9b2e6ecaf20a2b178daee2 |
| upstream/layers/Embed.py | 17e7c3577324c41a0da427a199c955b782fde905aabb1f7cbc3c4e15ebd4ae35 |
| upstream/layers/Conv_Blocks.py | 16d9f2d9e4fa094dc357901e32beecda9839709bcca02625e6447186933ce4e1 |
| upstream/layers/__init__.py | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| adapted/LICENSE | 8a6caa178ea3f33ebff5d7bb5558628cf5b423305dc30d3630f86564c7db94a2 |
| adapted/tn_models/TimesNet.py | b670bf680f7b8b2621581dc6563d5ab7f772f670e379c7f438b3e14a738a3e30 |
| adapted/tn_layers/Embed.py | 17e7c3577324c41a0da427a199c955b782fde905aabb1f7cbc3c4e15ebd4ae35 |
| adapted/tn_layers/Conv_Blocks.py | 16d9f2d9e4fa094dc357901e32beecda9839709bcca02625e6447186933ce4e1 |
| import_diff.patch | 75549585018be3dd72ee23139a429dba2c59c7d88604b8286eafd6ce18d01217 |

`adapted/tn_layers/{Embed,Conv_Blocks}.py` are byte-identical to upstream.

## Why all evaluation forwards run at batch size 1

`FFT_for_Period` selects the top-k periods from the **batch-mean amplitude
spectrum** (`frequency_list = abs(xf).mean(0).mean(-1)`, then topk). The
selected periods therefore depend on which windows happen to share a batch.
Averaging over identical copies of one window reproduces that window's own
spectrum, but any heterogeneous batch couples the windows' predictions. TN0
pins every evaluation forward (epoch-0 diagnostic, per-epoch validation, final
test, recheck replay) to batch size 1 so each window's period selection is a
function of that window alone; training keeps the protocol batch of 32. This
is fixed before any results are seen and disclosed in the final report.
