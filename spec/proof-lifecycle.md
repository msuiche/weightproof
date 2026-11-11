# Proof lifecycle, commitment renewal, and versioning

Status: normative target for the D11+ verifier and serving integration. The current
crate verifies stored key and request proofs; the serving API described here is
future work.

## 1. Proof-pending semantics

A response and its proof are different objects. A provider MAY return tokens before
the corresponding proof is available, but MUST label the response `proof-pending`.

A client MUST NOT treat `proof-pending` as evidence that the returned tokens were
produced by the committed model. It MAY use the tokens only as unverified provider
output. It MUST NOT cache them as verified, present them as verified to another
party, or mark the request complete.

The request states are:

| State | Meaning | Client action |
|---|---|---|
| `proof-pending` | Tokens are available; proving is unfinished or a proof is unavailable for retry. | Treat output as unverified; wait, retry, or reject according to local policy. |
| `verified` | The complete key/request proof and statement validate. | Accept output as verified only after checking the statement and certificate. |
| `rejected` | Parsing, statement, certificate, transcript, or proof validation failed. | Reject the output and record the named cause. |
| `unavailable` | No proof arrives within the caller's deadline or the provider reports a permanent proving failure. | Treat output as unverified and fail the request if verification is required. |

A verifier API MUST distinguish these outcomes. A command-line verifier MUST use
exit status 0 for `verified`, 1 for `rejected` with a named cause, and 2 for usage
errors. `proof-pending` is a protocol state, not a successful verification result.

Timeouts and retries MUST be bounded by the client. A retry MUST use the original
statement and a fresh proof, not a previously accepted transcript or a replayed
provider claim. A client MUST reject a proof for a different statement, batch, or
parameter set with a named error.

## 2. Commitment certification and model identity

A weight commitment authenticates only the weights covered by that commitment. It
does not by itself authenticate the model's identity, architecture, quantization,
or serving semantics. A provider's unsigned assertion is not a certificate.

A commitment certificate MUST bind:

- the weight commitment and its Merkle configuration;
- a human- or registry-readable model identifier and revision;
- model architecture and expert-routing semantics;
- quantization scales, rounding, overflow behavior, nonlinear tables, and decoding
  policy, including greedy decoding and tie-breaking;
- proof-system and parameter-set identifiers and versions;
- the first and last request numbers covered by the certificate; and
- expiration, renewal, and optional revocation information.

The client MUST obtain the expected model identity from an independent trust root:
a client-pinned value, a trusted registry, or a signed certificate/transparency
log. The client MUST verify that trust root before accepting any proof. A proof
for honestly committed but substituted weights MUST be rejected when the expected
identity differs.

The statement covered by verification MUST include:

`H(version ‖ commitment ‖ prompt ‖ cfg ‖ sampling ‖ tokens)`

Here `cfg` includes the architecture, quantization, code, and runtime parameters
needed to interpret the computation. `sampling` includes the decoding policy and
any seed. `tokens` identifies the checked range, not necessarily the whole reply.
The statement digest MUST also bind the parameter and shape digests used by the
proof system. A mismatch in any of these values is a different statement and MUST
fail closed.

## 3. Commitment renewal

FlashProve-style commitments have a finite use limit, approximately 16 million
requests. A certificate MUST state its exact limit. A provider MUST renew the
offline commitment before that limit is exhausted. A client MUST reject new proofs
past the certified range instead of assuming that the old commitment remains valid.

A renewal MUST create a fresh offline commitment and certificate, not merely extend
the old certificate's dates. The renewal certificate MUST bind the new commitment
to the same model identity and revision, unless the model itself is intentionally
changed. It MUST include a signed link to the preceding certificate and its request
range. The client MUST validate that link and the complete certificate chain back
to its trust root.

Renewal MUST NOT silently change architecture, quantization, decoding semantics,
proof-system parameters, or statement format. Any such change requires an explicit
new model revision and version transition. Clients MUST be told the new revision
before proofs under it are accepted.

A renewed commitment MAY be used only after its certificate is published and the
client has accepted the renewal. A provider MUST NOT reuse old proofs under a new
commitment. Every renewed request requires a proof tied to the new commitment and
statement.

## 4. Proof-format and parameter versioning

The statement envelope MUST contain a `version` field and separate identifiers for
proof format, code/parameter set, model revision, and commitment epoch. These
identifiers MUST NOT be inferred from weights or proof bytes alone.

A verifier MUST implement explicit version dispatch. Unknown versions, unsupported
feature bits, non-canonical encodings, truncated inputs, and trailing bytes MUST be
rejected with named errors. A verifier MUST NOT interpret a newer format as an
older one, accept a downgrade, or accept a proof under a different parameter set
than the statement names.

Backward compatibility is OPTIONAL and versioned. A verifier MAY accept multiple
versions only when each has an independent, reviewed parser and verifier path.
Shared code MUST NOT silently relax validation rules for older versions.

Version transitions MUST define:

- the first epoch or request number at which the new version is required;
- the last epoch or request number accepted under the old version;
- how clients discover supported versions and parameter digests; and
- how renewal preserves model identity while the format changes.

A parameter-file digest is not a format version. A commitment renewal is not a
format migration. Both MUST be represented explicitly in the statement and in the
client's acceptance rules.

## 5. Batched proofs and per-request extraction

For a batched proof, each client MUST be able to verify its own request without
receiving other clients' prompts or outputs. The extracted statement MUST bind the
client's own request and the shared commitment epoch. The client MUST NOT need to
trust the provider's batch-level assertion that all requests were covered.

A batch proof MUST make the shared point and the per-request statements explicit.
A verifier MUST check the selected request's inclusion and reject replay of a proof
for another request, another batch, another position, or another model identity.
Per-request extraction MUST preserve the same rejection and versioning rules as a
standalone proof.

## 6. Acceptance rule

A response is accepted only when all of the following hold:

1. The response is not `proof-pending` or `unavailable`.
2. The complete key/request proof verifies with the exact transcript.
3. The statement matches the client's request and parameter set.
4. The commitment certificate chains to the client's independent trust root.
5. The certificate covers the request number and has not expired or been revoked.
6. The model identity, architecture, quantization, decoding semantics, and versions
   match the client's expectations.

If verification is required and any condition fails, the client MUST reject the
output with a named cause. An unverified provider claim, a valid-looking proof for
another statement, or a proof for substituted weights is not an acceptance path.
