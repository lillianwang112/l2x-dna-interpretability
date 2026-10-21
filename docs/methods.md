# Methods and evaluation

## Synthetic data and split

The mentor-supplied simulator generates ancestors, inserts `AAAAA` into a subset, and mutates their descendants. Labels are determined by motif presence **after mutation**; mutations can create or remove a positive example. Shared ancestry produces correlated background context.

The 10,000 original sequence strings and their order are preserved. Sample and family IDs were recovered by replaying the seeded generator and matching every original row. A fixed seed-42 manifest partitions whole families:

| Partition | Families | Sequences | Positive | Negative |
| --- | ---: | ---: | ---: | ---: |
| Training | 60 | 6,000 | 2,320 | 3,680 |
| Validation | 20 | 2,000 | 686 | 1,314 |
| Test | 20 | 2,000 | 736 | 1,264 |

No ancestor family or identical DNA sequence crosses partitions. Both the teacher and explainer fit only training families. Validation supports diagnostic development; the final protocol fixes models, metrics, and illustrative sample IDs before test performance is computed. Generating held-out prediction arrays does not fit models on those rows.

## Architecture and training

| Component | Implementation |
| --- | --- |
| Input | One-hot DNA; selector shape `(N, 100, 4)` |
| Teacher | PyTorch Conv1d: 32 filters, width 10, ReLU, flattened linear head, sigmoid |
| Selector | Two Keras Conv1D layers: 32 filters, width 7, same padding, ReLU; width-1 scoring layer |
| Relaxed sampler | k Gumbel-perturbed categorical draws at temperature 0.5; positionwise maximum across draws |
| Predictor Q | Flattened masked DNA, Dense(256), Dense(128), two-class softmax |
| Objective | Categorical cross-entropy to the teacher’s two-class probabilities |

The teacher uses seed 0 and ten epochs. Original L2X models use k=1–10, seeds 0/1/2, and 15 epochs. Adam learning rate is 0.001; teacher batch size is 64, and L2X batch size is 128. All reported checkpoints use their final epoch.

The relaxed mask is not a set of exactly k unique unit weights. Repeated draws can overlap, and softmax spreads fractional weight across positions. The model’s internal evaluation threshold can retain more than k positions when scores tie. Reported evaluation instead uses a deterministic exact-k mask, breaking ties by lower position index. Frozen-checkpoint checks found no discrepancy between those rules in the diagnosed validation examples.

## Identity and provenance

Each teacher-training tensor retains its sample ID through shuffling. Prediction export compares canonical-order inference with an independent reverse-order inference pass restored to the original order. The recorded export matched all 10,000 rows exactly in the checked environment.

Teacher probabilities and selector scores use `.npz` artifacts with ordered sample IDs, dataset/split fingerprints, training indices, and a checksum over values and metadata. Selection scores also identify the teacher artifact. Loaders reject reordered IDs, changed values, different splits, and unverified legacy `.npy` files. These checks detect accidental mixing; they are not a guarantee against deliberately fabricated artifacts.

The [final protocol](../results/evaluation_protocol.json) locks source and artifact hashes. `evaluate_saved.py` verifies them before and after inference, restores all original models and both control arms, and checks restored selectors against the saved scores. It performs no optimization or checkpoint selection. The original training artifacts remain unchanged.

The public [evaluation JSON](../results/final_evaluation.json) contains all per-seed validation/test measurements, aggregate summaries, paired control differences, and the sequences/scores used in the example figure. Plotting reads this JSON without loading models. [Figure hashes](../figures/manifest.json) bind the exported PNG/PDF files to their input data and plotting script.

## Metric definitions

- **Any motif overlap:** fraction of positive sequences with at least one selected position inside any true motif occurrence.
- **Complete-window recovery:** fraction of positive sequences for which all five bases of at least one motif occurrence are selected. This is impossible for k<5.
- **Position precision:** selected motif positions divided by k, averaged over positive sequences.
- **Position recall:** selected motif positions divided by all true motif positions in a sequence, averaged over positives. Overlapping motif windows use the union of their positions.
- **Uniform-random overlap:** exact probability that a random k-subset intersects the union of motif positions, averaged over positives: `1 − C(L−m,k)/C(L,k)`. Seeded Monte Carlo checks are also retained.
- **Class agreement:** fraction of examples for which Q and the teacher have the same argmax class.
- **Balanced agreement:** mean of teacher-negative and teacher-positive agreement. It gives the teacher’s two predicted classes equal weight.
- **Teacher KL:** mean `KL(P_teacher || Q)` in nats. Probabilities are clipped at 1e-7 for logarithms. Lower indicates closer probability imitation.
- **Teacher cross-entropy and probability MAE:** supplementary measures of probability imitation, retained separately from class agreement.
- **True-label accuracy:** agreement with the synthetic motif-presence labels. It is distinct from imitation of an imperfect teacher.
- **Removal diagnostic:** Q evaluated on the complement of the same selected mask. This uses the approximator, not an intervention on the teacher.

The constant-class baseline selects the majority **teacher-predicted training class**, then evaluates that fixed prediction on held-out rows. Its balanced agreement is 50% when both teacher classes occur. The soft-probability baseline uses the mean teacher probabilities on training rows. The label-majority baseline separately uses the majority synthetic training label.

## Matched mask-training comparison

Each original k=5 checkpoint supplies one frozen selector and the same initial Q weights to both arms. Each arm receives a fresh Adam optimizer and 15 additional epochs with identical training rows and batch permutations. One uses the original stochastic soft masks; the other uses deterministic exact top-5 masks. Only Q is updated. The selector and teacher remain fixed.

Both arms are evaluated on exact top-5 inputs. Final-epoch metrics are reported for every seed. The [comparison figure](../figures/mask_comparison.png) shows paired seed measurements and means ± sample SD on validation and test families. Additional soft-mask evaluation draws from the validation control are retained in the [validation summary](../results/mask_training_validation.json) and generated run artifacts.

The control tests mask-distribution alignment for a fixed selector. It is not a newly trained selector or a general improvement to the L2X method. Mean test KL falls by 20.2%, while the direction of the balanced-agreement change differs between validation and test.

## Interpretation and limits

High any-overlap can coexist with low complete-window recovery and imperfect teacher imitation. A model can find part of the data-generating motif while missing features or behavior used by the teacher. The teacher itself has limited generalization to new families: test label accuracy is 68.50%, AUROC is 0.705, and positive-label recall is 214/736 (29.08%). A predictor that follows the true motif can disagree with this teacher.

Masking also changes the input distribution. Complement masks and all-input/all-zero diagnostics of Q do not establish causal necessity for the teacher. Mask alignment improves probability imitation in this experiment, but a substantial family-held-out gap remains.

All uncertainties describe three explainer seeds with one fixed teacher. They do not represent independently trained teachers, confidence intervals, or statistical significance. Transfer is evaluated under the same synthetic generator; no conclusions are established for experimental binding measurements, biological causality, other motifs, or transformer mechanistic interpretation.

## Reference

Chen, J., Song, L., Wainwright, M., & Jordan, M. (2018). [Learning to Explain: An Information-Theoretic Perspective on Model Interpretation](https://proceedings.mlr.press/v80/chen18j.html). *Proceedings of ICML*, PMLR 80, 883–892. Adaptation informed by the [original implementation](https://github.com/Jianbo-Lab/L2X).
