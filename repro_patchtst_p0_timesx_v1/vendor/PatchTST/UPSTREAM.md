# PatchTST vendor provenance

- upstream: https://github.com/yuqinie98/PatchTST (PatchTST_supervised subset)
- upstream_commit: 204c21efe0b39603ad6e2ca640ef5896646ab1a9
- upstream_commit_date: 2023-08-11T08:40:14+08:00
- upstream_subject: Update README.md
- retrieved_utc: 2026-10-03T03:50:19Z
- retrieval: git clone --quiet https://github.com/yuqinie98/PatchTST.git (local machine, then files copied verbatim)
- license: Apache License 2.0 (LICENSE copied verbatim into upstream/ and adapted/)
- adaptation: import-only package isolation (layers.->ptst_layers., see import_diff.patch); directory renames models/->ptst_models/, layers/->ptst_layers/; no other content change

## upstream file sha256 (pristine copies in upstream/)
```
c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4  upstream/LICENSE
49d8bb865e1226d6338842f4b33c4bc6992cefb048dc2a4a5a8c410ca708586b  upstream/models/PatchTST.py
df67173153787c2356bdfb6491159cd754332ef7382986efe879e1fbea8ebf26  upstream/layers/PatchTST_backbone.py
21c06c70a90c60ee2a269b5c600c702834dea22cdfd72915e6b0f8b4a28db3f6  upstream/layers/PatchTST_layers.py
e64c0ccded9228b347134e7368420d3fb10f70c75145b5ff2d8bdd8c3af59df6  upstream/layers/RevIN.py
```

## adapted file sha256 (runtime copies in adapted/)
```
c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4  adapted/LICENSE
6d439643d25abe4f1ecb7fadf93304b6ba1c22323aa88676c3ba111a461fe850  adapted/ptst_models/PatchTST.py
bb42b4d18b7893e9a78b6ec841c12a0fc266738a9bdcc6ec0b3371bd686c5b58  adapted/ptst_layers/PatchTST_backbone.py
21c06c70a90c60ee2a269b5c600c702834dea22cdfd72915e6b0f8b4a28db3f6  adapted/ptst_layers/PatchTST_layers.py
e64c0ccded9228b347134e7368420d3fb10f70c75145b5ff2d8bdd8c3af59df6  adapted/ptst_layers/RevIN.py
7f11a496fffc2941b39d8c0ebcbfbb42232acac2ab50d8dd2fe59b7f3a868b87  import_diff.patch
```
