# Bounded final literal-tool capture

This diagnostic follows the completed five-job final API batch. It does not change the frozen engine24f0, Tabby5a, recipeR1, model files, runtime defaults, or original test clients.

The first two clients run the original budgeted4 and unbudgeted4 requests exactly. All request payloads, response IDs, variants, rendered-prompt hashes, native/backend full output strings, source/configuration hashes, and owned cleanup must match. Capture integrity is independent of original semantic success; a valid raw capture of eight wrong arguments remains eight failed semantic tests.

Optional --include-thinking-off adds four separately named checks copied from the original unbudgeted4. Their only request change is the existing enable_thinking flag set to false. They retain the original Think briefly message and exact expected tool string. Their validator requires empty reasoning and the same correct single tool value. They do not substitute for, repair, or recategorize the original eight checks.

The observer captures only the eight original thinking-enabled requests. Its collector/finish hooks and site startup mechanism are unchanged from the independently reviewed strings observer; only exact request matching, bounds, source pins and metadata differ. The controller reuses the exact reviewed strings_only_controller.py server-only attachment and SignalGuard, plus final_api_controller.py configuration and the frozen owned lifecycle. No extra generation awaits, token/KV reads or output rewriting occur.

## CPU verification

Run tabbyapi-agent/.venv-tools/bin/python test_literal_capture_controller_cpu.py. Fourteen checks pass, covering original payload bytes, request/response-ID matching, exact source pins, raw capture completeness, unchanged original command lines, supplemental field-only derivation, exclusive output claims, and honest semantic versus capture counts. A complete synthetic raw-capture fixture built around the actual saved eight failed API records is accepted as a valid capture with eight semantic failures; an unexpected server exit is rejected. This is adapter verification, not a new model result.

The observer separately passed eight actual5a formatting-to-collector CPU fixtures plus13 negative request guards and cap/phase/prompt checks. Existing source review remains bounded: no additional model tests were run during preparation.

## Authorized host command

Only after the final five-job batch has completed, its PID has exited, every last owned server/client group is empty, port8899 is free and there is no remaining GPU compute process:

    python3 literal_capture_controller.py --job literal-final-305-24f0-5a-job.json --setup ACTUAL_COMPLETED_FINAL_SETUP_STATUS --output results/literal-final-305-24f0-5a --observer-dir literal-final-observer --include-thinking-off

The job is a copy of the completed final3.05 single job's six source/model/tuning fields, with a unique diagnostic label. It keeps one slot, RAM mode,262144cache,2048chunk,MTP5/.6,legacy precision guards andKSPLIT1. The configure adapter independently validates the exact completed setup and recipe HEAD before launching. The controller accepts no benchmark or extra client selection.

Exit0 means capture and every semantic check passed. Exit1 means capture is valid but one or more original/supplemental semantic checks failed. Exit2 means capture integrity/lifecycle could not be certified. All original final-batch reports remain immutable.
