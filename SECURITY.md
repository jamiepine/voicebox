# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email security details to: [security@voicebox.sh](mailto:security@voicebox.sh)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will:
- Acknowledge receipt within 48 hours
- Provide a timeline for addressing the issue
- Keep you informed of progress
- Credit you in the security advisory (if desired)

## Security Best Practices

### For Users

- **Keep Voicebox updated** - Updates include security patches
- **Verify downloads** - Only download from official releases
- **Local processing** - Voice data stays on your machine
- **Network security** - Use HTTPS when connecting to remote servers

### For Developers

- **Dependencies** - Keep all dependencies up to date
- **Code review** - All PRs require review before merging
- **Secrets** - Never commit API keys or signing keys
- **Signing** - All releases are cryptographically signed

## Known Security Considerations

### Local Processing

Voicebox processes all audio locally by default. Your voice data never leaves your machine unless you explicitly enable remote server mode.

### Remote Server Mode

When connecting to a remote server:
- Ensure the server is on a trusted network
- Use HTTPS for remote connections
- Verify server identity before connecting

### Auto-Updates

- Updates are cryptographically signed
- Signature verification happens before installation
- Only HTTPS endpoints are allowed

### Python Server

The embedded Python server:
- Runs locally by default (localhost only)
- Can be configured for remote access
- Uses standard FastAPI security practices

## Disclosure Timeline

- **Day 0**: Vulnerability reported
- **Day 1-2**: Initial assessment and acknowledgment
- **Day 3-7**: Investigation and fix development
- **Day 8-14**: Testing and release preparation
- **Day 15+**: Public disclosure (if applicable)

Timeline may vary based on severity and complexity.

## Security Updates

Security updates will be:
- Released as patch versions (e.g., 0.3.2)
- Documented in CHANGELOG.md
- Announced via GitHub releases
- Automatically delivered via auto-updater

---

Thank you for helping keep Voicebox secure! 🔒
