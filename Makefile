.PHONY: help install test lint format paper bundle clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies (editable + dev + analysis extras)
	pip install -e ".[dev,analysis]"
	@echo "Optional: download fastText langID for R5 reward"
	@echo "  wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P data/raw/"

test:  ## Run pytest (CI-safe subset, no GPU required)
	pytest tests/ -v --tb=short

test-r5:  ## Run R5 lang tests (requires data/raw/lid.176.bin)
	@test -f data/raw/lid.176.bin || (echo "Missing data/raw/lid.176.bin"; exit 1)
	pytest tests/test_rewards_lang.py -v

lint:  ## Run ruff linter
	ruff check src/ scripts/ data/ tests/

format:  ## Auto-format with ruff
	ruff format src/ scripts/ data/ tests/

paper:  ## Compile paper PDF (requires tectonic or pdflatex)
	cd paper && tectonic main.tex
	@echo "Output: paper/main.pdf"

bundle:  ## Build arXiv submission tar.gz
	cd paper && bash build_arxiv_bundle.sh
	@echo "Output: paper/arxiv_submission.tar.gz"

aggregate:  ## Aggregate eval JSONs → results/training/master.csv
	python -m src.analysis.aggregate \
	    --eval_dir results/training/eval/ \
	    --output results/training/master.csv

figures:  ## Regenerate paper figures from master.csv + training logs
	cd paper && python figures/make_figures.py \
	    --master ../results/training/master.csv \
	    --training_logs ../results/training/ \
	    --output figures/

preflight:  ## Run pre-training environment check
	bash scripts/preflight.sh

clean:  ## Remove build artifacts
	rm -rf paper/arxiv_submission paper/arxiv_submission.tar.gz
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc paper/*.bbl paper/*.blg
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache build dist *.egg-info
