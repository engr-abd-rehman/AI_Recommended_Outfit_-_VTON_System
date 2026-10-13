# Security Boundaries

FashionAI is a local research prototype. Bind the development server to
`127.0.0.1`, not a public interface. Do not expose it directly to the internet.

## Known Gaps

- There is no authentication, authorization, production session management,
  upload quota, rate limit, or persistent/distributed job queue.
- APIs accept server file paths; absolute-path confinement needs hardening
  before deployment beyond a trusted local operator.
- Original images, intermediate garments, and generated results are stored locally.
- Job state disappears on restart. The single-job guard is per-process, not
  protection across multiple Flask workers or machines.
- GPU calls and downloaded model code require a trusted execution environment.
- Upstream diffusion is invoked without a production safety-filter service.

Image-content validation and UUID filenames are useful, but do not establish
that the application is hardened against hostile or oversized uploads.

## Safe Local Use

Use permitted photos and provider downloads. Keep environments separate, verify
download checksums, and avoid concurrent generations. Do not put tokens, `.env`
files, personal photos, logs, or datasets into issues or commits. Ignore rules
are a safeguard, not a replacement for checking the exact staged file list.

Before any public deployment, add authentication, input/path confinement,
resource/time limits, isolated per-job storage, cancellation, appropriate
content controls, a persistent queue, and a real production server configuration.

Report security issues privately through the repository's available security
reporting channel or directly to the maintainer. Do not publish sensitive sample
inputs or credentials in a public issue.
