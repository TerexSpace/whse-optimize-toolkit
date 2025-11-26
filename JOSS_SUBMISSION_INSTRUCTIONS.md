# JOSS Submission Checklist and Instructions

## Journal of Open Source Software (JOSS) Submission Guide

This document provides step-by-step instructions for submitting SME-DT-ERP to JOSS.

---

## Pre-Submission Checklist

### 1. Software Requirements ✅

- [x] **Open source license**: MIT License included
- [x] **Substantial software**: ~1,400 LOC (exceeds 1,000 LOC minimum)
- [x] **Research application**: Digital twin simulation for warehouse management
- [x] **Feature complete**: Core simulation engine functional
- [x] **Installation documentation**: README.md with instructions
- [x] **Example usage**: Included in README and `run_simulation.py`
- [x] **Automated tests**: pytest suite with 679 lines of tests
- [x] **API documentation**: Docstrings in all public functions

### 2. Paper Requirements

- [x] **paper.md**: Markdown format paper (250-1000 words)
- [x] **paper.bib**: BibTeX references with DOIs
- [ ] **Author information**: Name, ORCID, affiliation (YOU MUST ADD)
- [x] **Summary section**: Describes software purpose
- [x] **Statement of Need**: Explains research application
- [x] **References**: 25 verified citations

### 3. Repository Requirements

- [ ] **GitHub repository**: Create public repository
- [ ] **GitHub Actions CI**: `.github/workflows/ci.yml` (included, needs activation)
- [ ] **Zenodo archive**: Archive release for DOI
- [ ] **Release tag**: Create v0.1.0 release

---

## Step-by-Step Submission Process

### Step 1: Complete Author Information

Edit `paper/paper.md` and replace placeholders:

```yaml
authors:
  - name: YOUR FULL NAME
    orcid: 0000-0000-0000-0000  # Get ORCID at https://orcid.org/
    affiliation: 1

affiliations:
  - name: Your University/Institution
    index: 1
```

### Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `sme-dt-erp`
3. Description: "Digital Twin Framework for ERP-Integrated Warehouse Management in SMEs"
4. Visibility: **Public** (required for JOSS)
5. Initialize with: Do NOT add README (we have one)
6. Create repository

Push code to GitHub:

```bash
cd sme_dt_erp
git init
git add .
git commit -m "Initial commit: SME-DT-ERP v0.1.0"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/sme-dt-erp.git
git push -u origin main
```

### Step 3: Verify CI/CD Works

1. After pushing, check GitHub Actions tab
2. Ensure all workflows pass (test, lint, build)
3. Fix any failing tests before proceeding

### Step 4: Create GitHub Release

1. Go to repository → Releases → "Create a new release"
2. Tag version: `v0.1.0`
3. Release title: `SME-DT-ERP v0.1.0`
4. Description:
   ```
   Initial release of SME-DT-ERP: Digital Twin Framework for 
   ERP-Integrated Warehouse Management in Small and Medium Enterprises.
   
   Features:
   - SimPy-based discrete-event simulation engine
   - ERP adapter interface with mock implementation
   - What-if scenario analysis
   - Automatic calibration from ERP event logs
   - Docker containerization support
   ```
5. Publish release

### Step 5: Archive on Zenodo

1. Go to https://zenodo.org/ and log in (use GitHub OAuth)
2. Link your GitHub repository: Settings → GitHub → Enable repository
3. Zenodo will automatically archive each GitHub release
4. Copy the Zenodo DOI (format: 10.5281/zenodo.XXXXXXX)

### Step 6: Update paper.md with Zenodo DOI

Add to paper.md front matter:

```yaml
repository: https://github.com/YOUR-USERNAME/sme-dt-erp
archive_doi: 10.5281/zenodo.XXXXXXX
```

### Step 7: Submit to JOSS

1. Go to https://joss.theoj.org/
2. Click "Submit" (requires GitHub login)
3. Enter repository URL: `https://github.com/YOUR-USERNAME/sme-dt-erp`
4. JOSS will:
   - Automatically detect `paper.md` and `paper.bib`
   - Run pre-review checks
   - Assign an editor

### Step 8: Respond to Review

JOSS uses open peer review on GitHub:

1. Editor will open a review issue
2. Reviewers will check against JOSS criteria
3. Respond to feedback within the review issue
4. Make requested changes and push updates
5. After approval, paper is published with DOI

---

## JOSS Review Criteria

Reviewers will evaluate:

| Criterion | Our Status |
|-----------|------------|
| **Substantial scholarly effort** | ✅ Novel digital twin calibration approach |
| **Quality of writing** | ✅ Clear, well-structured paper |
| **Functionality** | ✅ Core features work as documented |
| **Documentation** | ✅ README, docstrings, examples |
| **Tests** | ✅ pytest suite included |
| **Community guidelines** | ✅ CONTRIBUTING.md included |
| **License** | ✅ MIT (OSI-approved) |

---

## Required Actions Before Submission

### YOU MUST DO:

1. **Add your name and ORCID** to `paper/paper.md`
2. **Create GitHub repository** and push code
3. **Create release** (v0.1.0)
4. **Archive on Zenodo** and get DOI
5. **Update paper.md** with Zenodo DOI
6. **Submit** at https://joss.theoj.org/

### Optional Improvements:

- Add more ERP adapters (Odoo, ERPNext)
- Increase test coverage to 90%+
- Add Sphinx documentation site
- Create demo video

---

## Estimated Timeline

| Task | Time |
|------|------|
| Add author info | 5 minutes |
| Create GitHub repo | 10 minutes |
| Push code & verify CI | 15 minutes |
| Create release | 5 minutes |
| Zenodo archive | 10 minutes |
| Submit to JOSS | 10 minutes |
| **Total** | **~1 hour** |

| Review process | 4-8 weeks (varies) |

---

## Support

- JOSS documentation: https://joss.readthedocs.io/
- JOSS submission requirements: https://joss.theoj.org/about
- GitHub Actions docs: https://docs.github.com/actions
- Zenodo docs: https://help.zenodo.org/

---

## File Checklist

Ensure these files exist before submission:

```
sme_dt_erp/
├── paper/
│   ├── paper.md          ← JOSS paper (update author info!)
│   └── paper.bib         ← References with DOIs
├── .github/
│   └── workflows/
│       └── ci.yml        ← GitHub Actions CI
├── tests/
│   ├── __init__.py
│   └── test_core.py      ← Unit tests
├── core.py               ← Main implementation
├── __init__.py           ← Package init
├── setup.py              ← Installation config
├── requirements.txt      ← Dependencies
├── README.md             ← Documentation
├── LICENSE               ← MIT License
├── CONTRIBUTING.md       ← Contribution guidelines
├── Dockerfile            ← Container support
└── docker-compose.yml    ← Multi-service config
```

Good luck with your submission! 🎉
