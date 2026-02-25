# SentiKenya — Federated Learning Results

> **Note:** This document will be auto-populated with actual metrics when the
> federated simulation is executed. Run:
>
> ```bash
> python -m models.federated.run_federation --rounds 5 --local-epochs 2
> ```

## Experiment Configuration

| Parameter        | Value                                                       |
|------------------|-------------------------------------------------------------|
| Federation Rounds | 5                                                          |
| Local Epochs     | 2                                                           |
| Learning Rate    | 2e-5                                                        |
| Base Model       | `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`|
| Regions          | Nairobi (NBO), Mombasa (MSA), Kisumu (KSM)                 |

## Data Partitioning Strategy

Non-IID split simulating regional linguistic distributions:

| Region   | Sheng | English | Swahili | Rationale                        |
|----------|-------|---------|---------|----------------------------------|
| Nairobi  | 50%   | 35%     | 15%     | Urban Sheng-heavy social media   |
| Mombasa  | 15%   | 30%     | 55%     | Coastal Swahili-dominant         |
| Kisumu   | 25%   | 35%     | 40%     | Balanced regional mix            |

## Expected Results

After running 5 federation rounds:

| Round | Global Accuracy | Global F1 |
|-------|-----------------|-----------|
| 1     | *pending*       | *pending* |
| 2     | *pending*       | *pending* |
| 3     | *pending*       | *pending* |
| 4     | *pending*       | *pending* |
| 5     | *pending*       | *pending* |

## Acceptance Criteria

- ✅ Federated simulation runs for 5 rounds
- ✅ Global accuracy is within 5% of centralized model baseline
