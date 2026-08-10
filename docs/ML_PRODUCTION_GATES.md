# The Production Gate Framework for AI/ML Systems
### A research-backed checklist of the gates every real-world, production-grade ML/AI solution needs

*Compiled from: Google's "ML Test Score" paper (Breck et al., 2017), NIST AI Risk Management Framework, OWASP ML Security Top 10 and OWASP Top 10 for LLM Applications (2025), MLOps engineering blogs, Reddit/Medium/DEV.to practitioner post-mortems, and 2025–2026 industry failure-rate research (Gartner, RAND, MIT).*

---

## 0. Why This Framework Exists

Before the gates themselves, the numbers that justify building them:

- **80.3%** of enterprise AI projects fail to deliver their promised business value (RAND Corporation, 2025) — roughly double the failure rate of comparable non-AI IT projects.
- **95%** of generative AI pilots produce zero measurable P&L impact (MIT Project NANDA, 2025).
- **Only 28%** of AI use cases fully meet ROI expectations; **20% fail outright** (Gartner survey of 782 I&O leaders, late 2025).
- **85–91%** of deployed models degrade over time due to drift, and most models that do get built (**70–90%**) never reach production at all.
- The consistent root causes across every study: weak data foundations, no connection to a real business workflow, no governance, and prototypes that were never engineered for what happens *after* the demo.

This is the point of "gates": a gate is a **hard checkpoint** a model/system must pass — automatically or via sign-off — before it is allowed to move to the next stage. No gate, no promotion. This mirrors Google's own internal finding: teams that had *any* checklist, even an informal one, caught entire classes of failure that experienced engineers otherwise missed.

---

## 1. The 12 Gate Categories — Overview

| # | Gate Category | Core Question It Answers |
|---|---|---|
| 1 | Business & Problem-Framing | Should this even be an ML/AI solution, and how do we know it worked? |
| 2 | Data | Is the data trustworthy, governed, and fit for purpose? |
| 3 | Model Development | Is the model itself sound, tuned, and better than the alternative? |
| 4 | Responsible AI / Fairness | Does the model treat people and groups fairly? |
| 5 | ML Infrastructure & Testing | Is the pipeline reproducible, tested, and debuggable? |
| 6 | Security | Can the model/system be attacked, poisoned, stolen, or misused? |
| 7 | LLM/RAG-Specific Evaluation | (If generative) Is it grounded, non-hallucinating, and injection-resistant? |
| 8 | Deployment & Release | Can we ship this without betting the whole system on it? |
| 9 | Monitoring & Observability | Will we *know* the moment it breaks? |
| 10 | Governance, Compliance & Documentation | Can we prove — to auditors, regulators, and our future selves — what this system does? |
| 11 | Cost, Latency & Scalability | Does it meet its SLA at a cost the business can sustain? |
| 12 | Operational Readiness | Is there a human system (on-call, runbooks, rollback) around the technical one? |

Google's ML Test Score paper found something worth designing around: teams that scored well in three categories but neglected a fourth still had *systemically unreliable* systems. **A gate framework is only as strong as its weakest category** — so score/track each category independently, not just an averaged total.

---

## 2. Gate 1 — Business & Problem-Framing Gate
*(Fails here = the #1 cause of AI project death, per Gartner/RAND/MIT data above)*

| Check | Pass Criteria |
|---|---|
| Problem justified for ML | A simpler rules engine or heuristic was explicitly considered and rejected with a documented reason |
| Business metric defined | A single business metric (revenue, cost, churn, delinquency, etc.) is named, and its relationship to the ML metric (AUC, RMSE, F1) is explicit — not assumed |
| Success criteria set *before* build | Numeric target agreed with stakeholders in advance, not fitted after seeing results |
| Executive/workflow sponsorship | The system is integrated into an existing workflow, with a named owner, not a bolt-on side project |
| Baseline exists | Performance of "no model" / current manual process / a trivial baseline is measured for comparison |
| Go/no-go decision rights | It is explicit who can say "kill this" and at what checkpoint |

**Why it's a hard gate, not a soft one:** the research is unambiguous that projects disconnected from a workflow and an executive-owned metric fail regardless of model quality.

---

## 3. Gate 2 — Data Gates

Based on Google's ML Test Score "Data" tests (7 tests) plus modern data-quality tooling (Great Expectations, feature stores).

| Check | Pass Criteria |
|---|---|
| **Schema captured** | Feature expectations (types, ranges, distributions) encoded in a schema that can be checked automatically, not just known informally |
| **Feature value proven** | Each feature's incremental predictive value is measured (ablation, correlation) — no "kitchen sink" feature sets |
| **Feature cost bounded** | Latency, memory, and upstream-dependency cost of every feature is known and acceptable |
| **Meta-level compliance** | Prohibited features (protected attributes, deprecated sources) are programmatically blocked, not just avoided by convention |
| **Privacy controls in pipeline** | PII handling, access control, and right-to-deletion propagation are tested — not assumed because "we stripped names" |
| **Feature velocity** | A new feature can go from idea to production in a known, bounded time (weeks, not quarters) |
| **Feature code is unit-tested** | The code that generates features has real unit tests — bugs here are nearly invisible once in the pipeline |
| **Automated data-validation gate in CI** | Pipeline fails closed if incoming data breaks schema, distribution, or null-rate thresholds (e.g., via Great Expectations or TFX Data Validation) |
| **Data lineage tracked** | Every training run can be traced back to the exact data snapshot that produced it (DVC, lakeFS, or equivalent) |
| **No training/serving skew by construction** | Features are computed identically at train and serve time — ideally via a shared feature store (Feast, Tecton), since this mismatch is cited repeatedly as the single most common silent-failure cause |

---

## 4. Gate 3 — Model Development Gates

Based on Google's ML Test Score "Model" tests (7 tests).

| Check | Pass Criteria |
|---|---|
| **Code-reviewed & versioned** | Model spec is checked into a repo and reviewed like any other production code |
| **Offline↔online correlation known** | A small A/B test with a deliberately degraded model confirms that the offline metric actually predicts the online/business outcome |
| **Hyperparameters tuned, not defaulted** | Systematic search (grid, Bayesian) was run — not "the first config that worked" |
| **Staleness impact measured** | You know, quantitatively, how model quality degrades with model age, so a retraining cadence can be chosen deliberately |
| **Beats a simple baseline** | The production model is regularly re-compared to a trivial baseline (linear model, heuristic) to justify its complexity/cost |
| **Quality checked on slices, not just aggregate** | Performance is validated per important subgroup (country, device, customer segment) — a model can improve 1% globally while collapsing 50% on one slice |
| **Promotion thresholds are numeric and pre-declared** | e.g., "no worse than −1% AUC vs. current champion," codified in CI, not decided ad hoc at release time |

---

## 5. Gate 4 — Responsible AI / Fairness & Bias Gates

| Check | Pass Criteria |
|---|---|
| Protected-attribute correlation checked | Inputs are tested for strong correlation with protected classes (race, gender, age, etc.) even when those attributes aren't directly used |
| Fairness metric chosen and measured | At least one of demographic parity, equalized odds, predictive equality, or disparate impact ratio is computed per group |
| Bias-audit tooling used | Fairlearn, AIF360, or Aequitas run against the model with results documented (these are the three most-cited open-source toolkits in current practice) |
| Mitigation applied where needed | Reweighing, adversarial debiasing, or threshold calibration applied if a fairness gap is found — accepting the typical 1–5% accuracy trade-off is a documented decision, not an accident |
| Explainability available | SHAP or equivalent is available for any model influencing a consequential decision about a person |
| Human-appeal path exists | For high-stakes decisions (credit, hiring, healthcare), there is a way for an affected person to contest an outcome |
| Feedback-loop risk assessed | Check whether the model's own outputs will bias its future training data (e.g., a hiring model that deprioritizes a group, producing fewer "successful hires" from that group in next year's training set) |

---

## 6. Gate 5 — ML Infrastructure & Testing Gates

Based on Google's ML Test Score "Infra" tests (7 tests) — this is the category most teams skip, and the one whose absence causes the most 2 a.m. pages.

| Check | Pass Criteria |
|---|---|
| **Training is reproducible** | Same data + same code → same model (or documented sources of nondeterminism, e.g., seeded RNGs) |
| **Model spec unit-tested** | A fast test trains for one gradient step on random data to catch API-usage bugs before a full (expensive) training run |
| **Full pipeline integration-tested** | One automated test exercises data → features → train → validate → deploy end-to-end, on a schedule, not just per-PR |
| **Model quality gated before serving** | An automated system either blesses or vetoes a newly trained model — it cannot reach production without passing |
| **Model is debuggable** | You can feed one example through the model and inspect the computation step-by-step (critical for numerical-instability bugs) |
| **Canary before full serving** | Every new model version is tested against real production traffic *before* getting 100% of it |
| **Rollback is fast, safe, and rehearsed** | Rolling back is treated as an emergency procedure the team has actually practiced — not a theoretical capability |
| **CI/CD promotion gates codified** | Format/metadata validation, performance thresholds, compliance checks, and load tests all run automatically before a model gets a "champion" alias |

---

## 7. Gate 6 — Security Gates

Two separate threat surfaces exist: classical ML models, and (if applicable) LLM/generative systems. Both need dedicated gates — a web-app firewall does not defend against either.

### 7a. Classical ML — OWASP Machine Learning Security Top 10

| Risk | Gate / Mitigation |
|---|---|
| Adversarial input manipulation | Input validation + adversarial-robustness testing on any model exposed to untrusted input |
| ML supply-chain attacks | Pin and scan all third-party datasets, pretrained weights, and libraries; verify checksums/signatures on anything pulled from a public model hub |
| Model inversion / privacy leakage | Rate-limit query volume; add differential-privacy noise where the model touches sensitive training data |
| Data poisoning | Validate and version all training data; monitor for anomalous shifts in training-set statistics |
| Model theft / extraction | Authenticate and monitor API access; watch for the query patterns typical of extraction attacks |
| Transfer-learning attacks | Audit any pretrained/foundation model before fine-tuning on top of it |
| Model poisoning (registry-level) | Hash-verify and sign models pulled from the registry; restrict who can push to "Production" alias |

### 7b. LLM / GenAI systems — OWASP Top 10 for LLM Applications (2025)

| Risk | Minimum Gate |
|---|---|
| **LLM01 Prompt Injection** (direct + indirect via RAG/tool content) | Input scanning, instruction hierarchy, treat all retrieved/tool content as untrusted, sandbox tool access |
| **LLM02 Sensitive Information Disclosure** | Scan outputs for PII/secrets, strip secrets from prompts, apply DLP |
| **LLM03 Supply Chain** | Scan model files, verify provenance, audit plugins/dependencies |
| **LLM04 Data & Model Poisoning** | Validate fine-tuning data, monitor for backdoor triggers post-training |
| **LLM05 Improper Output Handling** | Never pass raw LLM output into a shell, SQL query, or renderer without validation/sanitization — this is the same injection class OWASP catalogued for web apps 20 years ago, now with an LLM in front of it |
| **LLM06 Excessive Agency** | Least-privilege tool access, human confirmation for irreversible actions (send, delete, publish), full tool-call logging |
| **LLM07 System Prompt Leakage** | Design prompts assuming they *will* be extracted; never embed secrets in them |
| **LLM08 Vector/Embedding Weaknesses** | Secure the vector DB, sanitize ingested documents before embedding |
| **LLM09 Misinformation** (hallucination) | See Gate 7 below — this needs its own evaluation pipeline |
| **LLM10 Unbounded Consumption** | Rate limits, token/cost caps, timeouts, to prevent resource-exhaustion/denial-of-wallet attacks |

**For agents specifically**, add pre-execution "safety gates" — a lightweight preflight layer checked before *any* tool call fires: intent/scope check, permission check, data-policy check, dry-run for irreversible actions, and a rate limit.

---

## 8. Gate 7 — LLM/RAG-Specific Evaluation Gates
*(Skip this section if your system is classical/non-generative ML)*

| Check | Pass Criteria |
|---|---|
| Retrieval evaluated separately from generation | Recall@K, MRR, and context relevance measured independently of generation quality — a system can have perfect retrieval and still hallucinate, or vice versa |
| Faithfulness/groundedness measured | The answer contains only what the retrieved context supports (RAGAS "faithfulness" or equivalent) |
| Answer relevance measured | The answer actually addresses the question asked |
| Golden/regression dataset exists | A curated set of test queries with expected answers/sources runs on every PR — every real production failure becomes a permanent regression test |
| LLM-as-judge calibrated against humans | If using an LLM judge to scale evaluation, its scores are validated against human labels first (judges have known biases: verbosity inflation, self-preference, position effects) |
| Citation/source accuracy checked | Cited sources actually support the claims placed next to them |
| Continuous production sampling | Evaluation isn't just a pre-deploy gate — a sample of live traffic is scored continuously, since corpus growth and model updates silently shift quality over time |
| Cost/latency/quality frontier documented | You know the trade-off curve for your specific system (e.g., chunk size vs. hallucination rate, reranker choice vs. latency) rather than tuning blind |

---

## 9. Gate 8 — Deployment & Release Gates

| Check | Pass Criteria |
|---|---|
| Deployment strategy chosen deliberately | Shadow (zero user risk, validates function/latency, no business-impact signal) → Canary (small % of real traffic, real signal, slower) → full rollout — chosen based on actual risk tolerance, not habit |
| Guardrail metrics set *before* seeing canary data | Thresholds decided in advance (the "guardrail threshold anti-pattern" is setting them after looking at results) |
| Champion/challenger held with headroom | Champion model keeps ~20% serving headroom so a bad challenger can be rolled back without a capacity scramble |
| Blue/green used only where appropriate | Blue/green assumes the new version is instantly verifiable — true for stateless web code, often **false** for ML models, where quality problems surface only after real traffic and time; canary/shadow is usually the safer default for model changes specifically |
| Automatic rollback trigger wired | A CRITICAL guardrail breach pages on-call and/or auto-reverts — it isn't a manual, "someone notices" process |
| Model/serving-binary compatibility checked | New model artifacts are tested to actually *load* into the current serving binaries before going live (a classic mismatch source when training code changes faster than serving code) |

---

## 10. Gate 9 — Monitoring & Observability Gates

Based on Google's ML Test Score "Monitor" tests (7 tests) — the category that catches what pre-deployment testing structurally cannot.

| Check | Pass Criteria |
|---|---|
| Dependency-change alerts | Team is subscribed to changes in every upstream data source it depends on |
| Data invariants monitored live | Serving-time input is checked against the same schema as training data, continuously |
| Training/serving skew monitored | The single most production-incident-causing, least-implemented test, per Google's own internal survey — log a sample of serving traffic and diff feature values against training-time computation |
| Model staleness monitored | Age of the production model (and any dependent aggregation tables) is tracked and alertable |
| Numerical stability monitored | NaNs/infinities in weights or activations trigger alerts, not silent corruption |
| Compute-performance regression monitored | Latency, throughput, and RAM usage tracked by model/data version, not just in aggregate |
| **Prediction-quality regression monitored** | Statistical bias in predictions, real-time label comparison where available, and periodic human-annotated spot checks — this is what actually tells you the model is still working, not just still running |
| Drift detection wired to retraining | Feature drift (PSI, KS test) and prediction-distribution shift are tracked as leading indicators — don't wait for labeled ground truth, which can lag by days or weeks |
| On-call dashboards exist | A single place shows champion vs. challenger latency, error rate, prediction distribution, and the primary business metric in real time |

---

## 11. Gate 10 — Governance, Compliance & Documentation Gates

| Check | Pass Criteria |
|---|---|
| NIST AI RMF functions addressed | **Govern** (roles, policies, approval gates) → **Map** (system scope, stakeholders, data lineage) → **Measure** (evaluations, red-teaming, thresholds) → **Manage** (risk register, incident response) — even informally, all four are covered |
| Maximum-acceptable-risk threshold set in advance | A pre-declared "we will not deploy above this risk level" line, decided before deployment pressure exists |
| Model Card produced | Documents intended use, training data summary, evaluation results (including per-slice/per-group), and known limitations |
| Datasheet for training data produced | Documents provenance, collection process, labeling method, and known biases of the dataset (Gebru et al. framework) |
| AI System Card produced (if user-facing) | Documents the full deployed system — model + retrieval layer + safety filters + human-oversight mechanism — not just the model in isolation |
| Audit trail maintained | Who trained, approved, and deployed each model version is logged and queryable |
| Regulatory mapping done | If applicable: EU AI Act risk tier identified; sector rules (HIPAA, GLBA, FCRA, etc.) mapped to specific controls, not left implicit |
| Role-based access control | Who can trigger training, who can approve promotion, and who can access production data are all explicitly defined — not "everyone on the data team" |

---

## 12. Gate 11 — Cost, Latency & Scalability Gates

| Check | Pass Criteria |
|---|---|
| Latency SLA defined at p95/p99, not average | Tail latency, not mean latency, is what users and downstream systems actually feel |
| Capacity sized for peak, not average | Provisioning based on 95th/99th-percentile request rate, since latency degrades non-linearly as concurrency rises |
| Load-tested before launch | Synthetic load tests validate the SLA holds under realistic peak traffic, including during a canary rollout when old + new versions run concurrently |
| Cost per request/token tracked and tied to a product metric | Not just "what does the GPU bill say" — cost is connected to a unit economics number the business cares about |
| Autoscaling policy tuned on the right signal | Queue depth and batch size, not raw GPU utilization alone (which tends to overprovision) |
| Graceful degradation path exists | A defined fallback (cached response, simpler model, "system busy" state) if the primary model/service is overloaded or down |

---

## 13. Gate 12 — Operational Readiness Gates

| Check | Pass Criteria |
|---|---|
| Definition of done includes ops, not just accuracy | Monitoring, documentation, runbooks, and a rollback plan are part of "done" — a good offline metric alone does not qualify a model for production |
| On-call rotation exists with playbooks | Named rotation, with runbooks for the common failure modes (upstream data source down, feature drift, need-to-rollback) |
| Escalation path is unambiguous | No guessing who gets paged when something breaks at 2 a.m. |
| Post-incident review process exists | Failures produce a documented postmortem that feeds back into the gates above (a bad slice becomes a Gate-3 test; a skew incident becomes a Gate-9 alert) |
| Rollback has been rehearsed, not just built | Teams that only discover their rollback process during a real incident report it being "so painful they'd never do it again" — practice it when nothing is on fire |

---

## 14. Master Consolidated Checklist (Quick-Reference)

Use this as your at-a-glance production sign-off sheet. Full detail for each item is in the sections above.

- [ ] **Business:** ML justified over simpler approach; business metric ↔ ML metric link proven; baseline measured; sponsor named
- [ ] **Data:** Schema + automated validation gate; feature value/cost measured; privacy controls tested; lineage tracked; feature-store or equivalent prevents train/serve skew
- [ ] **Model:** Code-reviewed; offline↔online correlation validated; hyperparameters tuned; staleness impact known; beats baseline; slice-level quality checked; numeric promotion thresholds codified
- [ ] **Fairness:** Protected-attribute correlation checked; fairness metric computed (Fairlearn/AIF360/Aequitas); explainability available; feedback-loop risk assessed
- [ ] **Infrastructure:** Training reproducible; model spec unit-tested; full pipeline integration-tested; model debuggable; canary required before full serving; rollback rehearsed
- [ ] **Security (classical):** Adversarial-input testing; supply-chain scanning; theft/inversion monitoring; poisoning defenses
- [ ] **Security (LLM/GenAI):** Prompt-injection defenses; output validated before downstream use; least-privilege tool access; PII/secret scanning; rate/cost limits
- [ ] **LLM Evaluation (if generative):** Faithfulness + relevance measured; golden regression set in CI; LLM-judge calibrated; continuous production sampling
- [ ] **Deployment:** Shadow/canary strategy chosen deliberately; guardrails set before seeing data; automatic rollback wired; serving-binary compatibility checked
- [ ] **Monitoring:** Data invariants, staleness, skew, numerical stability, compute performance, and prediction-quality regression all alertable; drift feeds retraining
- [ ] **Governance:** Model card + datasheet produced; audit trail maintained; regulatory tier mapped; RBAC enforced
- [ ] **Cost/Scale:** p95/p99 SLA defined; load-tested at peak; cost-per-request tied to a business metric; graceful degradation exists
- [ ] **Operations:** On-call + runbooks exist; escalation path clear; postmortem process feeds back into gates above

---

## 15. Scoring Your System (ML Test Score-Inspired Rubric)

Adapt Google's scoring approach for your paper's evaluation methodology:

- **0.5 point** per check: executed manually, with results documented
- **1 point** per check: automated, running repeatedly without manual intervention
- Sum points **within each of the 12 categories separately**
- **Your system's overall score = the MINIMUM of the 12 category scores** (not the average)

This last rule is the important one, and it's deliberate: a system with excellent monitoring but zero fairness testing is not "pretty good" — it is exactly as production-ready as its weakest gate. Reporting a minimum, not an average, is what stops a strong infrastructure score from masking a missing governance or security gate in your paper's results.

| Score Range | Interpretation |
|---|---|
| 0 | Research project, not a production system |
| (0, 1] | Untested; likely serious reliability holes |
| (1, 2] | First-pass productionization; more investment needed |
| (2, 3] | Reasonably tested; more could be automated |
| (3, 5] | Strong automated testing/monitoring; appropriate for mission-critical systems |
| > 5 | Exceptional |

---

## Sources & Further Reading

- Breck, Cai, Nielsen, Salib, Sculley — *"The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction"*, Google/IEEE Big Data 2017 — research.google.com/pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf
- Sculley et al. — *"Machine Learning: The High-Interest Credit Card of Technical Debt"*, NeurIPS 2014 workshop (foundational paper on ML technical debt)
- NIST AI Risk Management Framework (AI RMF 1.0) and companion Playbook — nist.gov
- OWASP Machine Learning Security Top 10 — owasp.org/www-project-machine-learning-security-top-10
- OWASP Top 10 for LLM Applications (2025) — owasp.org / genai.owasp.org
- Mitchell et al. — *"Model Cards for Model Reporting"*; Gebru et al. — *"Datasheets for Datasets"*
- Fairlearn, AIF360 (IBM), Aequitas (U. Chicago Center for Data Science and Public Policy) — open-source fairness toolkits
- RAGAS framework for RAG evaluation (faithfulness, answer relevance, context precision/recall)
- Gartner — *"AI Projects in I&O Stall Ahead of Meaningful ROI Returns"* (April 2026); RAND Corporation (2025); MIT Project NANDA (2025)
- Practitioner sources: MLOps engineering blogs (MLflow, Databricks, Snowflake, Galileo), DEV.to and Medium post-mortems on production ML failures, r/MLOps and r/MachineLearning community discussion threads

*Note: this document paraphrases and synthesizes publicly available research and industry writing rather than reproducing any single source verbatim — use it as a framework map, and pull the primary papers above directly for your paper's formal citations.*
