# Parent-only launch

Verify shared GPU ownership has been drained, both supervisors inactive and Qwen still disabled. Use the clean EOS-only fd8 runtime and frozen recipe254. This controller never enables a service, installs packages or changes source/models.

```sh
python3 /home/cruzspark/qwen-followup-20261009/measurement/eos-default-tool-validation/controller.py --recipe /home/cruzspark/qwen-followup-20261009/recipe-gpu-lock --runtime /home/cruzspark/qwen-followup-20261009/eos-candidate-runtime --contract /home/cruzspark/qwen-followup-20261009/measurement/eos-default-tool-validation/runtime-contract.json --contract-sha256 9825b80310e39b3656e314b39076109c1231857e74f20469a65d1d4c8a8ac4cb --model-inputs /home/cruzspark/qwen-followup-20261009/measurement/eos-default-tool-validation/model-inputs.json --packages-sha256 4f71ec141cedfa82781135669cb3492ad1547ace7dca61d370fc2acc9fa0210a --output /home/cruzspark/qwen-followup-20261009/results/eos-default-tool28-fd8 --ready-timeout 300 --client-timeout 1800
```

No observer environment is admitted. Read-only setup/import preflight precedes the one owned server. The client deadline is1800seconds; retain failures and use a new output directory for any explicitly scheduled repetition.
