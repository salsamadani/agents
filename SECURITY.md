# Security Policy

## Reporting a vulnerability

Please do not report security problems in public issues or pull requests.

Report them privately instead: open the **Security** tab of this repository and click **Report a vulnerability**. Only the maintainers can see the report.

Include:

- what the problem is and which files or functions it affects
- steps to reproduce it
- the impact you expect, such as leaked API keys or code execution

We aim to reply within 7 days. Please give us a reasonable time to fix the problem before you disclose it publicly.

## Handling API keys

This project reads API keys such as `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` from the environment or from an `.env` file. Never commit real keys. If a key is exposed, revoke it with the provider right away.
