# Parent-only launch

Both supervisors must be inactive and their backend/GPU ownership drained by the parent. The Qwen unit remains disabled. Verify source/package/model/manifest bytes and both runtime checkouts before launch. The controller performs read-only setup/import checks, then starts only its own server; its server acquires the persistent shared lock. It refuses conflicting owners and never enables/restarts an existing service.

```sh
python3 /home/cruzspark/qwen-followup-20261009/measurement/literal-feature-eos-generation/controller.py --recipe /home/cruzspark/qwen-followup-20261009/recipe-gpu-lock --runtime /home/cruzspark/qwen-followup-20261009/literal-eos-candidate-runtime --plan /home/cruzspark/qwen-followup-20261009/measurement/literal-feature-eos-generation/plan.json --inputs /home/cruzspark/qwen-followup-20261009/measurement/literal-feature-eos-generation/prepared-inputs.json --inputs-sha256 6a47b8afa81e387ab8d07aeca30d78afc461f646b6671bfebed05130797afae3 --binding-source-sha256 687ba31ff103415a49f3474d13a9aee6734a8405b9dc063dc384a0f05bb1209c --model-inputs /home/cruzspark/qwen-followup-20261009/measurement/literal-feature-eos-generation/model-inputs.json --observer-dir /home/cruzspark/qwen-followup-20261009/literal-feature-eos-observer --observer-manifest-sha256 a45a4684524f2eb45614ae8f70808337c90c78bce7b0f7992fbde2c6bea7b865 --packages-sha256 4f71ec141cedfa82781135669cb3492ad1547ace7dca61d370fc2acc9fa0210a --output /home/cruzspark/qwen-followup-20261009/results/literal-feature-eos-generation-f4aadf --ready-timeout 300 --client-timeout 1800
```

Expected bounded duration is several minutes; the client deadline is1800seconds. Capture/semantic failures and cleanup records remain in the fresh output. Do not retry into an existing directory.
