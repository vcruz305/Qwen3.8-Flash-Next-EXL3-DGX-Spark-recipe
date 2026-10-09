# Retry after preflight-only framework rejection

The first source-frozen cdb40d3f launch exited during CPU-only import preflight. It rejected the installed Ubuntu `sitecustomize` solely because the module existed. No server or API client was launched. Its original bundle and nonterminal preflight result remain unchanged outside this retry directory.

This retry changes only the harness controller and its regression tests. It permits absence of customization or the exact verified Ubuntu module `/usr/lib/python3.12/sitecustomize.py`, resolving to `/etc/python3.12/sitecustomize.py`, SHA256 `43d81125d92376b1a69d53a71126a041cc9a18d8080e92dea0a2ae23be138b1e`. Root verified the package-owned apport exception-handler conffile; the admission record is included. Loaded usercustomize, observer environment, other module paths, changed contents, and changed symlink resolution remain rejected. Admitted bytes are rechecked before and after the clients. No diagnostic generation hook is used.

An outer failure recorder now marks failed preflight attempts terminal while retaining source/setup fields. It records zero chat POSTs and no launched server only while the state is preflight. Existing process registration, signal deferral, ownership verification and cleanup are unchanged. The revised 25-test CPU log is `harness-cpu-r2.log`; copied earlier review logs still refer to their explicit predecessor source hashes.

Client, final plan, native/expanded ID preparation, prepared input binding, source/runtime/package pins, request order and all 89 logical checks / 93 planned chat POSTs are unchanged. `plan.json` remains authoritative; the copied source-plan-generator.py records the original preparation and does not recreate the final cache-prefix revision.
