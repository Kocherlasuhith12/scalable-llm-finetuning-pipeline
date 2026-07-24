# 05. Data Pipeline

## Data Pipeline Stages
1. **Ingestion & Loader**: File parser supporting JSONL, CSV, and Parquet formats. Converts raw records into normalized prompt-response pairs (`instruction`, `input`, `output`).
2. **Quality Cleaning**: Text normalization, stripping control codes, removing HTML tags, unicode standardizing, and min/max length filtering.
3. **Deduplication Engine**:
   - **Exact Matching**: MD5/SHA256 hash sets.
   - **Fuzzy Deduplication**: MinHash LSH (Locality Sensitive Hashing) with configurable Jaccard similarity threshold.
4. **Preference Dataset Processing**: Formats pairwise preference data (`prompt`, `chosen`, `rejected`) required for DPO (Direct Preference Optimization).
5. **Splitting & Persistence**: Divides datasets into Train (80%), Validation (10%), and Test (10%) splits, persisting formatted artifacts for training.
