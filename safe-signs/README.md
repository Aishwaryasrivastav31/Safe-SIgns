# Safe Signs

When perception is fooled, can logic keep the decision safe?

## The question

A traffic-sign classifier can be pushed into reading the wrong sign by noise too small for a person to notice, or by a sticker. If the car acts on that reading directly, a stop sign can become "speed limit 120".

A car is not only a camera. It also knows things about where it is: whether the road is urban, rural or a motorway, whether there is a junction ahead, whether it is in a school zone, which speed limit is in force. Many sign readings are impossible in a given place. There is no 120 km/h limit in a town, and no stop sign without a junction.

This project combines a neural network that reads signs with a small logic layer that knows these rules, and asks:

1. How often does each system make an **unsafe** decision, one that lets the car go faster than the real sign allows, under different attacks?
2. Does the benefit come from the **logic**, or simply from being **cautious when unsure**?
3. Where does the logic fail? Specifically: against an attacker who knows the rules, and when the logic's own knowledge (the map) is wrong.
4. Can the safety properties of the decision layer be **verified exhaustively**?

## Contribution

A **rule-aware decision layer with a decision-level safety certificate**. The network may be fooled by any perception attack of any size; the decision layer still guarantees, by construction and verified exhaustively, that the car never acts on a sign the rules call impossible in the map context and never drives faster than the fastest rule-consistent sign (the **rule envelope**). For every image this gives a worst-case bound on overspeed that holds whatever the network outputs, and certifies some images as safe against every perception attack. The study then measures exactly where the guarantee stops: attacks that stay inside the rule envelope, and attacks on the map itself.

## System

**Perception.** A convolutional network written from scratch in NumPy (two convolution blocks, two dense layers), with forward and backward passes, Adam, and temperature scaling for calibrated probabilities. Gradients are checked against finite differences in the tests.

**Knowledge base.** Ten named rules over the map context, such as:
- speed limits must fit the road type,
- school zones only post 20 or 30 km/h,
- stop, yield, right-of-way and roundabout signs only appear at junctions,
- "end of 80" needs an 80 km/h limit in force,
- there are no pedestrian or cycle crossings on motorways.

Every rejection comes with a plain-English reason. The same rules are exported as a Prolog program (`rules.pl`).

**Decision layer.** Four systems are compared:

| System | What it does |
|---|---|
| network alone | acts on the most probable sign |
| network + fallback | same, but when the top probability is below a threshold, takes the most restrictive plausible sign |
| logic only | acts on the most probable sign that the rules allow |
| logic + fallback | both: the full neuro-symbolic system |

The middle two are ablations. They separate the effect of the rules from the effect of simple caution.

## Attacks

All attacks are white-box on the network:

- FGSM and PGD (L-infinity, 8/255),
- an unbounded 8x8 sticker (patch PGD),
- **dangerous**: targeted PGD towards the sign that would let the car go fastest,
- **rule-aware**: the same, but only towards signs the rules accept. This is the attack designed to defeat the logic layer.

## Evaluation

- Unsafe-decision rate, accuracy, fallback rate and mean overspeed, for every attack and every system.
- Paired comparisons of the full system with each ablation: bootstrap CI and exact McNemar test.
- **Wrong maps**: the context the system sees is corrupted for 10% or 20% of images, while the truth is unchanged.
- **Rule violations as an alarm**: how well the probability the network puts on impossible signs separates attacked images from clean ones (AUROC).
- **Exhaustive verification** of the decision layer: every valid context combined with every confident output and every split between two signs, about 72,000 cases per system. Four properties are checked, and counter-examples are recorded:
  - P1: never act on an impossible sign
  - P2: never exceed the legal maximum speed
  - P3: stop when unsure at a junction
  - P4: be as cautious as every plausible sign when unsure

## Results on GTSRB (first round, simulated map contexts)

Setup: the CNN was trained on 23,976 images from the GTSRB training set (26,640 images; 10% held out for validation and temperature scaling). Clean test accuracy is 93.6% on all 12,630 test images. Attacks and decisions were evaluated on 3,000 test images. Full tables are in `results/main/summary_gtsrb.txt`.

| Attack | Network alone | Network + fallback | Logic only | Logic + fallback |
|---|---|---|---|---|
| none (clean) | 2.1% | 1.3% | 1.5% | **1.0%** |
| FGSM | 28.2% | 26.5% | 22.1% | **16.8%** |
| PGD | 31.3% | 31.2% | 25.8% | **20.8%** |
| 8x8 sticker | 34.1% | 34.1% | 29.5% | **22.2%** |
| dangerous (targeted) | 62.4% | 60.9% | 16.9% | **3.9%** |
| rule-aware (targeted) | 52.4% | 50.5% | 52.3% | 50.5% |

*Share of images with an unsafe decision, meaning the car drives faster than the real sign allows.*

1. **The full system reduces unsafe decisions under every untargeted attack:** by 10 to 12 percentage points against FGSM, PGD and the sticker. Paired McNemar p < 0.001 against every ablation.
2. **Caution alone does not help against strong attacks.** Attacked images make the network confidently wrong: the fallback triggers on only 0.9% of PGD images. The rules expose the error, and the fallback then acts on it.
3. **Against the dangerous attack, unsafe decisions drop from 62.4% to 3.9%**, at the cost of a cautious fallback on 67% of images.
4. **The logic gives almost no protection against a rule-aware attacker.** The full system and network + fallback both reach 50.5% unsafe decisions; the difference is 0.07 points, with p = 0.5. Rules only catch errors that break them.
5. **Wrong maps raise the error rate, but the full system stays the safest.** With 20% of map facts wrong, clean-image unsafe decisions rise from 1.0% to 2.6%, still below the network alone (3.2%). Under PGD the full system stays near 21%.
6. **Rule violations work as an alarm for crude attacks only.** The probability given to impossible signs detects the dangerous attack with AUROC 0.86, but not the others (0.54 to 0.61).
7. **Verification:** across 71,896 context and output combinations, both logic-based systems violate none of the four properties. The network alone acts on an impossible sign in 21,702 of them and exceeds the legal speed in 6,760.

## Second round: adversarial training, real maps, threshold sweep

Same 3,000 test images; unsafe-decision rates. Map contexts from OpenStreetMap (Berlin, Potsdam) for 1,466 of the 3,000 images (15 classes with at least 5 real signs); the rest keep simulated contexts. In real contexts the true sign breaks at least one rule 2.1% of the time.

**Adversarial training and logic stop different attacks.** A second CNN was trained with PGD adversarial training (PGD-3, eps 8/255, half of each batch; clean accuracy 86.6% vs 93.6%).

| Attack | Standard network | Adv. trained network | Standard + logic + fallback | Adv. trained + logic + fallback |
|---|---|---|---|---|
| PGD | 32.8% | 22.1% | 19.8% | **9.9%** |
| FGSM | 29.9% | 18.8% | 16.2% | **6.8%** |
| Dangerous (targeted) | 62.9% | 29.3% | 5.0% | **4.4%** |
| Rule-aware (targeted) | 52.0% | **26.4%** | 51.2% | 20.7% |

Logic wins against the dangerous attack, adversarial training wins against the rule-aware attack the logic cannot see, and the combination is lowest or close to lowest everywhere.

**The fallback threshold does not buy safety.** Over tau from 0.3 to 0.95, network + fallback stays at 32-33% unsafe under PGD (the attacked network is confidently wrong, so it rarely falls back), and logic + fallback stays at 19-20%. Under the dangerous attack the fallback rate of logic + fallback never drops below 63%. The protection comes from the rules, not the threshold.

**Real maps cost something on clean images.** With real contexts, logic + fallback is unsafe on 2.53% of clean images vs 1.30% for network + fallback (McNemar p < 0.001); with simulated contexts it was 1.0%. With 20% wrong map facts it reaches 3.63%, above the network alone (3.10%), so the earlier result that the full system stays safer under map noise does not hold on real maps.

**Rule reliability on real maps.** The junction rule (stop, yield and right-of-way signs only at junctions) is broken by 11 of 52 real signs (21%); the school-zone rule by 1 of 21; the other rules with data by none. Four rules had no real signs to check.

## Third round: soft rules, junctions from road topology, more cities, training seeds

Code for these is in place and tested; results come from the next Colab run.

1. **Soft rules.** Instead of deleting a sign that breaks a rule, the rule multiplies its probability by the rule's smoothed violation rate on real maps, (violations + 20 x 0.01) / (real signs + 20). Rules real maps never break stay effectively hard (speed-fits-road weight 0.0014); the junction rule becomes soft (weight about 0.16). A sign that breaks only the junction rule can then still be acted on when the network is very sure, which targets the clean-image cost above. On the exhaustive check with the round-two reliabilities, soft logic + fallback has 0 violations of P2 (never above the legal maximum), P3 and P4, and 8,554 P1 cases, which it allows by design (acting on a sign that breaks an unreliable rule, e.g. stopping for a stop sign the map places away from a junction).
2. **Junctions from road topology.** A junction is now an OSM node shared by two different roads (or a roundabout) within 30 m of the sign, instead of two differently named roads nearby. This tests whether the 21% junction-rule violation is real or an artefact of the old approximation.
3. **More cities.** Berlin, Potsdam, München, Hamburg and a rural district (Landkreis Märkisch-Oderland), so more classes get real contexts.
4. **Training seeds.** The standard CNN is retrained with seeds 1 and 2 on the same images and contexts; `seeds.csv` reports mean and standard deviation of each system's unsafe rate over the three seeds.

## Fourth round: formal guarantee, stronger attacks, failure analysis

Code is in place and tested; results come from the next Colab run (the same run as round three).

### Safety properties

Let c be the map context the decision layer sees, p any probability vector from the network, d(p, c) the decision and v(d) its speed. Let A(c) be the signs the rules allow in c, and r(k, c) the speed sign k asks for.

| | Property | Formal statement |
|---|---|---|
| P1 | never act on an impossible sign | d(p, c) is in A(c), or is the safe default |
| P2 | never above the legal maximum | v(d) <= vmax(road(c)) |
| P3 | stop when unsure at a junction | fallback and junction(c) imply v(d) = 0 |
| P4 | as cautious as every plausible sign when unsure | fallback implies v(d) <= min r(k, c) over plausible k in A(c) |
| P5 | **rule envelope** | v(d) <= E(c) = max over k in A(c) of r(k, c), for every p |

**Theorem (P5).** For the hard-logic systems (logic only, logic + fallback), P5 holds for every probability vector p. *Proof.* The decision is either r(k, c) for a sign k in A(c), the minimum of such speeds (fallback), or the safe default. Signs with no rule (for example "priority road") are always in A(c) and ask for the current speed, so E(c) >= current speed >= safe default. Every branch is therefore at most E(c). The bound is checked on every valid context with the two output families of the exhaustive check plus 1,000 random probability vectors per context (Dirichlet at three concentrations and sparse vectors).

**Corollary (certificate).** If E(c) is not faster than the real sign allows, no perception attack of any size can make a hard-logic system unsafe on that image. The run reports the certified share of test images, the worst-case overspeed bound with and without rules, and checks on every attack that no certified image is ever unsafe. The certificate assumes the map is correct; the map-spoofing attack below measures what happens when it is not.

### Stronger attacks

| Attack | What the attacker knows and controls |
|---|---|
| **Rule-aware adaptive** | The network, the rules and the fallback. For each image it tries the three fastest signs the rules accept that would be unsafe, runs a 30-step targeted PGD towards each (so the network is confident enough to avoid the fallback), and keeps the first image on which the full system's decision is actually unsafe. |
| **Map spoof** | One fact of the map context (for example a spoofed road type or a removed junction), chosen to widen the rule envelope the most. Tested alone on clean images and combined with the rule-aware adaptive attack. |
| **Attack strength curve** | PGD and the rule-aware attack at eps = 2, 4, 8 and 16 / 255. |

These run on 600 test images against both the standard and the adversarially trained network.

### Baselines and metrics

- **Baselines in one table:** vanilla CNN, adversarially trained CNN, CNN + confidence threshold, the full system, and adversarially trained CNN + full system; unsafe rate and accuracy for every attack.
- **Safety-property violations on the test images:** how often each system acts on an impossible sign (P1) or exceeds the legal maximum (P2), per attack.
- **Clean vs robust trade-off:** clean accuracy, PGD accuracy, clean and worst-case unsafe rate, and clean fallback rate per model and system.

### Failure analysis

Every unsafe decision of the full system is assigned one cause:

1. **inside the rule envelope:** the network was fooled into a sign the rules accept but that is faster than the real one. Rules cannot catch this by design.
2. **fallback still too fast:** the system was unsure, but the real sign was not among the plausible ones.
3. **true sign breaks a rule:** the real map context contradicts the rules (real OpenStreetMap data), so the true sign was rejected.
4. **map wrong:** the context the system saw was wrong or spoofed.

The run also credits each image the rules saved (network alone unsafe, full system safe) to the rule that rejected the network's reading, and breaks the unsafe rate down by road type, junction and sign kind.

### Smoke test

On synthetic signs (500 images) the whole pipeline runs in about 10 minutes on a CPU. The certificate held on every attack: 0 unsafe decisions on certified images and 0 decisions above the bound. The envelope check found 0 violations for the logic systems and 7,847 for the network alone in 79,496 cases. These synthetic numbers only show that the code works.

## Running it

CPU only; no deep learning framework is needed.

```
pip install -r requirements.txt
python scripts/run.py --smoke      # synthetic signs, about 3 minutes
python -m pytest -q
```

For GTSRB, put the training images (and optionally the official test images with `GT-final_test.csv`) under `data/gtsrb/`. The loader accepts the official, torchvision and Kaggle folder layouts. `notebooks/run_colab.ipynb` downloads the data and runs everything. Then:

```
python scripts/run.py
python scripts/run.py --reuse-weights    # re-run the analysis without retraining
python scripts/run.py --reuse-weights --no-seeds   # also skip the two extra training seeds
```

Training the NumPy CNN on all of GTSRB takes a while on a CPU. Set `max_per_class` in `configs/default.yaml` to train on a subset.

## Layout

```
src/ss/signs.py        the 43 classes and what each asks the driver to do
src/ss/logic.py        knowledge base, explanations, decision layer, Prolog export
src/ss/cnn.py          NumPy CNN, Adam, temperature scaling
src/ss/attacks.py      FGSM, PGD, patch and targeted attacks
src/ss/contexts.py     map contexts for each image, and wrong-map corruption
src/ss/verify.py       exhaustive check of the decision layer, envelope check (P5)
src/ss/adaptive.py     rule-aware adaptive attack, map spoofing, attack strength curve
src/ss/analysis.py     baselines, property violations, trade-off, certificate, failure analysis
src/ss/data.py         GTSRB loading; synthetic signs for tests
src/ss/experiment.py   the full study and statistics
src/ss/report.py       figures and summary
```

## Limitations

- Map contexts are simulated unless `data/osm_contexts.json` exists. With OpenStreetMap they are real for the sign class but not for the exact photo, and OSM tags are incomplete.
- The rules are a small hand-written set for German signs. They cover a subset of real constraints.
- The verification covers the decision layer, not the network. P1 to P4 are checked over an abstraction of network outputs (confident, or split between two signs); P5 is proved for all outputs and checked on random ones.
- The certificate holds only for the map context the system sees. A wrong or spoofed map moves the envelope, which the map-spoof attack measures.
- The attacks are white-box and digital; physical stickers on real signs behave differently.
- Results on synthetic signs only show that the code works.
