# Migration Artifact

The temporary remote branch `artifact-transfer-20261109` stores a split copy of
`Topic1_transfer_20261109.zip` so that a new computer can restore the generated
datasets, validation keys, result tables, and checkpoints that are normally
excluded by `.gitignore`. The archive is split into ordinary Git blobs below
GitHub's 100 MB per-file limit because this repository has no available Git LFS
budget.

The archive contains:

- the full-day 5-minute RaySTEC parquet and metadata;
- the full-day IRI simulation parquet and metadata;
- the full-day NeQuick-G simulation parquet and metadata;
- all historical files under `results/tables/`;
- all historical files under `results/checkpoints/`;
- the handoff document and an offline Git bundle.

Expected archive properties:

```text
Size: 447707330 bytes
SHA-256: 623509581E8758DFE07BF4AAF921E3F850BD552BD718B6BDD1BDBA60F63320EB
```

After cloning or pulling the repository on the destination computer, fetch and
temporarily switch to the transfer branch:

```powershell
git pull origin main
git fetch origin artifact-transfer-20261109
git switch -c artifact-transfer-20261109 `
  --track origin/artifact-transfer-20261109
.\scripts\restore_transferred_artifacts.ps1
git switch main
git branch -D artifact-transfer-20261109
```

The restore script concatenates the ordered chunks into a temporary ZIP,
verifies SHA-256, and extracts only `datasets/` and `results/`. It does not
overwrite source code. The extracted files remain ignored by Git. The transfer
branch is intended for one computer migration and should be deleted from GitHub
after the destination confirms the restored hashes and file counts.
