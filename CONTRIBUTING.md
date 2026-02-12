# Contributing to PII/PHI Classifier

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/pii-phi-classifier.git
   cd pii-phi-classifier
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

5. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Style

We follow PEP 8 style guidelines. Run these tools before committing:

```bash
# Format code
black src/ tests/

# Check linting
pylint src/

# Type checking
mypy src/
```

### Testing

Write tests for new features:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

Aim for >80% code coverage for new code.

### Documentation

- Add docstrings to all public functions and classes
- Follow Google docstring format
- Update README.md if adding new features
- Add inline comments for complex logic

Example:
```python
def classify_field(self, field_metadata: Dict, field_profile: Dict) -> ClassificationResult:
    """
    Classify a single field using PaLM 2
    
    Args:
        field_metadata: Field information (name, type, description, etc.)
        field_profile: Statistical profile (patterns, cardinality, etc.)
        
    Returns:
        ClassificationResult with category, confidence, and reasoning
        
    Raises:
        ValueError: If field_metadata is missing required keys
    """
```

## Pull Request Process

1. **Update tests**: Add or update tests for your changes
2. **Run tests**: Ensure all tests pass
3. **Update documentation**: Update README and docstrings
4. **Commit messages**: Use clear, descriptive commit messages
   ```
   Add email pattern detection to field profiler
   
   - Implemented regex-based email detection
   - Added confidence scoring
   - Updated tests
   ```

5. **Submit PR**: 
   - Describe what changed and why
   - Reference any related issues
   - Request review from maintainers

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines (black, pylint)
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts

## Reporting Issues

When reporting bugs or requesting features:

1. **Search existing issues** first
2. **Use issue templates** if available
3. **Provide details**:
   - Environment (OS, Python version, GCP project)
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and logs
   - Screenshots if applicable

## Code Review Guidelines

### For Reviewers

- Be respectful and constructive
- Focus on code quality and maintainability
- Suggest improvements, don't demand perfection
- Approve once issues are resolved

### For Contributors

- Respond to feedback promptly
- Don't take criticism personally
- Ask questions if feedback is unclear
- Update the PR based on review comments

## Project Structure

```
pii-phi-classifier/
├── src/
│   ├── discovery/      # Schema discovery
│   ├── profiling/      # Field profiling
│   ├── classification/ # AI classification
│   ├── training/       # Model training
│   ├── masking/        # Data masking
│   └── utils/          # Utilities
├── tests/              # Unit tests
├── config/             # Configuration
└── docs/               # Documentation
```

## Development Tips

### Local Testing

Use DirectRunner for local Beam pipeline testing:
```python
pipeline_options = PipelineOptions(
    runner='DirectRunner'  # Instead of DataflowRunner
)
```

### Mock GCP Services

Use mocks for unit tests:
```python
from unittest.mock import Mock, patch

@patch('google.cloud.bigquery.Client')
def test_function(mock_bq):
    mock_bq.return_value.query.return_value = []
    # Your test code
```

### Environment Variables

Use `.env` file for local development:
```bash
GCP_PROJECT_ID=test-project
GCP_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

## Getting Help

- 💬 **Slack**: #pii-phi-classifier
- 📧 **Email**: data-governance@example.com
- 📚 **Documentation**: [Link]
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-org/pii-phi-classifier/issues)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
