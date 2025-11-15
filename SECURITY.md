# Security Policy

## Supported Versions

Currently, this project is in active development. Security updates will be provided for the latest version.

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### How to Report

1. **DO NOT** open a public GitHub issue
2. Email the details to `mail@sanjibsen.com` with subject "Security Vulnerability in weblatex AI Text Detection"
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Assessment**: We will assess the vulnerability and provide an initial response within 5 business days
- **Updates**: You will receive updates on the progress toward a fix
- **Credit**: If you wish, we will credit you in the security advisory

## Security Best Practices

When deploying the AI Text Detection system:

1. **Authentication**: Always add authentication before exposing the API publicly
2. **HTTPS**: Use HTTPS in production environments
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **Model Security**: Store trained models securely
5. **Dependencies**: Regularly update dependencies to patch known vulnerabilities
6. **Docker Images**: Use official, updated base images and scan for vulnerabilities
7. **Input Validation**: While the system includes input validation, always sanitize user inputs
8. **Monitoring**: Monitor for unusual patterns or abuse

## Known Security Considerations

### AI/ML Security

- **Model Adversarial Attacks**: The models may be vulnerable to adversarial inputs designed to fool detection
- **Data Poisoning**: Training on malicious data could compromise model integrity
- **Model Extraction**: API access could potentially be used to reverse-engineer the model
- **Privacy**: Be aware of sensitive information in training data

### Recommendations

- Regularly retrain models with validated data
- Implement monitoring for unusual prediction patterns
- Consider adding adversarial robustness techniques
- Review training data for potential biases or poisoning

## Security Audit History

- **2024-11**: Initial CodeQL scan - No vulnerabilities found
- **2024-11**: Manual security review - No issues identified

## Security Features

Current security measures in place:

- ✅ Input validation via Pydantic
- ✅ No hardcoded credentials
- ✅ Safe file operations
- ✅ Error handling without information leakage
- ✅ Minimal Docker base image
- ✅ No SQL injection vectors (no database)
- ✅ CodeQL continuous scanning via GitHub Actions

## Future Security Enhancements

Planned security improvements:

- [ ] API authentication/authorization system
- [ ] Rate limiting middleware
- [ ] Request/response logging for audit
- [ ] Model integrity verification
- [ ] Encrypted model storage
- [ ] Security headers in HTTP responses
- [ ] Input sanitization hardening

## Contact

For security concerns: `mail@sanjibsen.com`

For general inquiries: Open a GitHub issue
