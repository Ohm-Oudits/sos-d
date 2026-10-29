# Security Policy

SOS parses hostile ELF/SBF bytes, RPC responses, transaction history, account
data, and optional model output. Treat the tool as security-sensitive analysis
software and run it with least privilege, especially when enabling replay or a
local model subprocess.

## Reporting a vulnerability

Please do not disclose an exploitable parser, sandbox escape, credential leak,
or remote execution issue in a public issue. Until a private repository security
channel is configured, contact the project maintainers privately and include:

- affected commit or release;
- operating system and toolchain versions;
- the smallest reproducer or fixture that demonstrates the issue;
- expected and observed behavior; and
- whether any credentials or external systems may be exposed.

If the report contains sensitive bytes, redact keys, tokens, private account
data, and client-confidential programs before transmission. Do not submit a
wallet seed, signing key, or model-provider credential.

## Scope

Reports involving ELF parsing, loader-state parsing, RPC response handling,
model-output validation, subprocess isolation, replay state provenance, or
unsafe report rendering are in scope. A detector's conservative finding or a
false positive without an underlying security impact is generally a quality
issue rather than a security vulnerability.

The project does not submit transactions or perform live-chain writes in its
core workflow.
