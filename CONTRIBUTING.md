# 🤝 Contributing to Fabgaurd-AI

Thank you for your interest in contributing to **Fabgaurd-AI**! This document provides guidelines and best practices for contributing to the project.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of:
- Experience level
- Gender identity and expression
- Sexual orientation
- Disability
- Personal appearance
- Race and ethnicity
- Age
- Religion or lack thereof

### Expected Behavior

- Be respectful and considerate in communication
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, trolling, or discriminatory language
- Personal attacks or insults
- Publishing others' private information
- Spam or off-topic discussions
- Any conduct that could reasonably be considered inappropriate

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.8 or higher
- Git installed and configured
- GitHub account
- Basic understanding of PyTorch and computer vision (for code contributions)

### Finding Issues to Work On

1. **Browse Open Issues**: Check the [Issues](https://github.com/adarshshkla/Fabgaurd-AI/issues) page
2. **Look for Labels**:
   - `good first issue`: Perfect for beginners
   - `help wanted`: Community input needed
   - `bug`: Something isn't working
   - `enhancement`: New feature or request
   - `documentation`: Improvements or additions to documentation

3. **Ask Questions**: If an issue is unclear, ask for clarification in the comments

---

## How to Contribute

### Ways to Contribute

#### 1. Report Bugs

Found a bug? Please create an issue with:

- **Clear title**: Descriptive summary of the bug
- **Environment**: OS, Python version, PyTorch version
- **Steps to reproduce**: Detailed steps to trigger the bug
- **Expected behavior**: What should have happened
- **Actual behavior**: What actually happened
- **Error messages**: Full stack trace if applicable
- **Screenshots**: If relevant

**Example Bug Report**:
```markdown
**Title**: Training crashes with CUDA out of memory error

**Environment**:
- OS: Ubuntu 20.04
- Python: 3.9
- PyTorch: 2.0.1+cu118
- GPU: NVIDIA RTX 3060 (6GB)

**Steps to Reproduce**:
1. Run `python train_teacher.py` with default settings
2. Training starts normally
3. Crashes at epoch 2, batch 45

**Error Message**:
```
RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB
```

**Expected**: Training should complete without memory errors
**Actual**: Crashes after ~1.5 epochs
```

#### 2. Suggest Enhancements

Have an idea? Create an issue describing:

- **Problem**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought about
- **Additional context**: Any relevant information

#### 3. Improve Documentation

Documentation improvements are always welcome:

- Fix typos or grammatical errors
- Clarify confusing sections
- Add examples or tutorials
- Improve code comments
- Translate documentation (future)

#### 4. Write Code

Contribute bug fixes, new features, or optimizations:

- Follow the [Pull Request Process](#pull-request-process)
- Adhere to [Coding Standards](#coding-standards)
- Include tests for new features
- Update documentation as needed

---

## Development Setup

### 1. Fork the Repository

Click the "Fork" button on [GitHub](https://github.com/adarshshkla/Fabgaurd-AI)

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/Fabgaurd-AI.git
cd Fabgaurd-AI
```

### 3. Add Upstream Remote

```bash
git remote add upstream https://github.com/adarshshkla/Fabgaurd-AI.git
```

### 4. Create Virtual Environment

```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch naming conventions**:
- `feature/`: New features
- `fix/`: Bug fixes
- `docs/`: Documentation changes
- `refactor/`: Code refactoring
- `test/`: Test additions/modifications

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

#### Code Formatting

```python
# Good
def train_model(model, dataloader, optimizer, epochs=30):
    """
    Train the model for specified number of epochs.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer instance
        epochs: Number of training epochs (default: 30)
    
    Returns:
        Trained model
    """
    for epoch in range(epochs):
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Training code here
            pass
    
    return model

# Bad
def trainModel(model,dataloader,optimizer,epochs=30):
    for epoch in range(epochs):
        for batch_idx,(inputs,labels) in enumerate(dataloader):
            pass
    return model
```

#### Naming Conventions

- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private**: `_leading_underscore`

```python
# Good
class DefectDetector:
    MAX_BATCH_SIZE = 64
    
    def __init__(self):
        self._model = None
    
    def predict_batch(self, images):
        pass

# Bad
class defect_detector:
    maxBatchSize = 64
    
    def __init__(self):
        self.Model = None
    
    def PredictBatch(self, images):
        pass
```

#### Docstrings

Use Google-style docstrings:

```python
def compute_accuracy(predictions, targets):
    """
    Compute classification accuracy.
    
    Args:
        predictions (torch.Tensor): Model predictions (N, C)
        targets (torch.Tensor): Ground truth labels (N,)
    
    Returns:
        float: Accuracy score between 0 and 1
    
    Raises:
        ValueError: If shapes don't match
    
    Example:
        >>> preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        >>> targets = torch.tensor([1, 0])
        >>> compute_accuracy(preds, targets)
        1.0
    """
    if predictions.size(0) != targets.size(0):
        raise ValueError("Batch sizes don't match")
    
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)
```

#### Imports

Organize imports in this order:

```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Local
from fabgaurd.models import EfficientNetTeacher
from fabgaurd.utils import load_config
```

#### Comments

```python
# Good: Explain WHY, not WHAT
# Use temperature scaling to soften probability distributions
# This helps knowledge distillation by revealing class similarities
soft_targets = F.softmax(logits / temperature, dim=1)

# Bad: Obvious comment
# Apply softmax to logits
soft_targets = F.softmax(logits, dim=1)
```

### Type Hints

Use type hints for function signatures:

```python
from typing import List, Dict, Optional, Tuple

def process_images(
    image_paths: List[str],
    batch_size: int = 32,
    device: str = 'cuda'
) -> Dict[str, List[float]]:
    """Process images and return results."""
    pass
```

---

## Testing Guidelines

### Writing Tests

Create tests in `tests/` directory:

```python
# tests/test_detector.py
import unittest
import torch
from fabgaurd import DefectDetector

class TestDefectDetector(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests"""
        cls.detector = DefectDetector('models/teacher_b0_refined.pth')
    
    def test_predict_returns_dict(self):
        """Test that predict returns correct structure"""
        result = self.detector.predict('test/Good/image.jpg')
        
        self.assertIsInstance(result, dict)
        self.assertIn('predicted_class', result)
        self.assertIn('confidence', result)
        self.assertIn('defect_detected', result)
    
    def test_confidence_in_range(self):
        """Test that confidence is between 0 and 1"""
        result = self.detector.predict('test/Good/image.jpg')
        
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_batch_predict(self):
        """Test batch prediction"""
        paths = ['test/Good/img1.jpg', 'test/Crack/img2.jpg']
        results = self.detector.predict_batch(paths)
        
        self.assertEqual(len(results), len(paths))

if __name__ == '__main__':
    unittest.main()
```

### Running Tests

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test
python -m unittest tests.test_detector.TestDefectDetector

# Run with verbose output
python -m unittest discover tests/ -v
```

---

## Pull Request Process

### 1. Update Your Fork

Keep your fork synchronized:

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

### 2. Create Feature Branch

```bash
git checkout -b feature/awesome-feature
```

### 3. Make Changes

- Write clean, documented code
- Follow coding standards
- Add tests for new features
- Update documentation

### 4. Commit Changes

Write meaningful commit messages:

```bash
git add .
git commit -m "Add defect localization feature

- Implement bounding box detection using Grad-CAM
- Add visualization utilities
- Update documentation with examples
- Add unit tests for new functionality"
```

**Commit message format**:
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed description (wrap at 72 chars)
- Reference issues: "Fixes #123" or "Relates to #456"

### 5. Push to Your Fork

```bash
git push origin feature/awesome-feature
```

### 6. Create Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill out the PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring

## Testing
- [ ] All existing tests pass
- [ ] Added new tests for new features
- [ ] Manually tested on [describe environment]

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-reviewed code
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated

## Related Issues
Fixes #123
```

### 7. Address Review Feedback

- Respond to comments
- Make requested changes
- Push updates to your branch

```bash
# Make changes
git add .
git commit -m "Address review feedback"
git push origin feature/awesome-feature
```

### 8. Merge

Once approved, a maintainer will merge your PR!

---

## Review Process

### What We Look For

- **Correctness**: Does the code work as intended?
- **Quality**: Is the code clean, readable, and maintainable?
- **Tests**: Are there adequate tests?
- **Documentation**: Is it well-documented?
- **Style**: Does it follow project conventions?

### Timeline

- **Initial review**: Within 1-3 days
- **Follow-up**: Ongoing until approval
- **Merge**: After 1+ maintainer approval

---

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General discussions and questions
- **Pull Requests**: Code reviews and contributions

### Getting Help

- Check existing [documentation](docs/)
- Search [closed issues](https://github.com/adarshshkla/Fabgaurd-AI/issues?q=is%3Aissue+is%3Aclosed)
- Ask in [Discussions](https://github.com/adarshshkla/Fabgaurd-AI/discussions)
- Tag maintainers in issues/PRs if urgent

---

## Recognition

Contributors will be:
- Listed in the project README
- Credited in release notes
- Acknowledged in the community

---

## Questions?

Don't hesitate to ask! We're here to help.

- **Issues**: For bug reports and features
- **Discussions**: For questions and ideas

---

**Thank you for contributing to Fabgaurd-AI! 🛡️**

*Together, we're guarding silicon, one die at a time.*

---

*Last Updated: February 2026*
