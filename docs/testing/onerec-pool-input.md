# OneRec pooled input test

This test-only path checks whether a single model forward actually consumes an
`input_ids` view backed by the pool. It is not a production framework adapter.
Run only within an authorized scope and confirmed device window, with a separate
external watchdog and resource observation. The companion [bounded Ascend
procedure](ascend-bounded-run.md) describes the preceding mapping and health
checks; those checks do not approve or launch the model.

The operator supplies `FLAGMEMORY_ONEREC_ENABLE=approved-single-forward` and
`FLAGMEMORY_ONEREC_GATE` pointing to a JSON execution record. Missing or invalid
fields fail before framework import. The record must contain:

- `model_id`, `revision`, `model_path`, `weights_confirmed`, and `files`: an
  independently checked SHA-256 manifest covering configuration, tokenizer,
  safetensors index, every shard referenced by that index, and all other files
  in the directory. Symlinks are rejected. The mounted model directory must
  be read-only, verified via filesystem mount flags. Existing complete weights are required;
  this test does not download or copy them.
- `mapping` and `health`, each with `path` and `sha256` referring to successful
  prior stage JSON; `image_id`, `probe_source_sha256`, `source_commit`,
  `device`, and `physical_device` bind the relevant execution identities.
  The source digest here refers to the probe evidence, which can be from an
  earlier fixed source snapshot than the model-test commit.
- `resource_scope` set to `onerec-single-forward`, `resource_approved` true,
  `host_memory_bytes`, `device_observation_limit_bytes`, `timeout_s`, and an
  `approved_resources` object containing exactly those same three values;
  `model_mount_read_only` and `external_watchdog_confirmed` must be true.
  Also specify timezone-aware `expires_at`, the three framework version fields
  (`torch_version`, `torch_npu_version`, `transformers_version`), and finite
  nonnegative `atol` and `rtol`. These attestations must reflect the separately
  verified actual container limits, read-only mount, and watchdog. This helper
  does not itself establish them or enforce a hardware memory quota.

The test uses one model on one mapped device, batch one and at most 128 tokens,
with cache disabled. It compares one ordinary forward with one pooled forward,
checks finite logits, tolerances and matching argmax, and verifies the embedding
hook reads the actual pooled storage and byte range. The model parameters,
activations and outputs are separate from the input pool; its observed device
memory limit is an observation line, not an HBM hard quota. The input pool
uses an 8 MiB tier and at most 16 MiB cumulative backing allocations; ordinary
input and model resources are accounted for separately. It does not manage KV,
run generation, training, or assess recommendation quality.

The caller removes hooks and stops using all external aliases before returning a
lease. The helper cannot discover or revoke arbitrary Python aliases. It records
and polls completion before return; a pre-return event failure retains the live
lease. Once return is attempted, failure makes the result uncertain and the
helper prohibits consumption or another return. Pool destruction reports any
pending, quarantined or live blocks and keeps unsafe backing under pool custody.
See the [test adapter](../../tests/pool/ascend_test_adapter.py) for its limited
candidate backend capabilities.
