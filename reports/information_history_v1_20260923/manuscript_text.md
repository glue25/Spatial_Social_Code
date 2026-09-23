# Suggested manuscript text (bounded claims)

## Controlled information-history diagnostic

We conducted a fixed-record diagnostic on 40 constructed base episodes, each with
four information histories and four readout conditions (640 final-answer calls).
The physical event sequence and terminal object locations were held fixed within
each base, while the target agent either witnessed the relevant change, did not
acquire relevant new information, received a relevant report, or was the intended
recipient of an undelivered report. Entity vocabularies were drawn from locally
held-out EnactToM task episodes; the original simulator trajectories and camera
visibility were not replayed. A separate set of ten constructed episodes was used
for engineering checks and excluded from formal evaluation.

The conditions were a linked SSC diagnostic readout, a deterministic lossless
prose serialization, removal of explicit cross-layer links, and removal of holder
and source-agent attribution. All conditions used the same provider-reported
Gemini-3-Flash-preview model and generation limits. Every formal call, including
11 invalid outputs, remained in the denominator. The diagnostic uses a new
factorized adapter over the existing StateStore and should not be conflated with
the historical online planner's exact input serialization.

Linked SSC achieved 100.0% correct world-state preservation across history pairs
and 97.5% correct target-state changes. The prespecified paired joint
correctness measure was 95.0%, compared with 92.5% for lossless prose and 0.0% for
both deletion conditions. Paired base-episode bootstrap intervals for the SSC–
prose difference included zero (+2.5 percentage points; 95% CI [-5.0,10.0]). The
two co-primary deletion comparisons were +95.0 points each (Bonferroni 97.5%
marginal CIs [87.5,100.0]). These deletion effects must be interpreted as sensitivity
to removed information: some histories become indistinguishable without links,
and unknown is then a defensible answer. They do not establish superiority over
equal-information representations. A stricter correct-state-and-exact-physical-
citation metric favored prose (90.0% versus 83.1%).

SSC used 354,498 total provider-reported tokens versus 455,731 for prose (22.2% fewer)
across 160 formal calls per condition. This is a descriptive cost comparison under
equal generation caps, not an exactly token-matched test or a dollar-cost estimate.
Together, the results demonstrate controlled use of acquisition and attribution
information in this text-state setting, while leaving code syntax superiority,
visual compilation fidelity and general online-memory advantages unestablished.
The small number of shared history templates further limits generalization.

## Statistical figure caption

**Fixed-world information-history diagnostic.** All 40 constructed bases and all
four readouts are included (160 calls per readout). Panels report correct world-
state preservation, correct target-state change, and paired joint correctness on
predefined contrasts. Invalid responses count as incorrect. Error bars show 95%
marginal percentile intervals from 10,000 base-episode bootstrap resamples; these
are distinct from the 97.5% intervals for the two co-primary paired differences.
Deletion conditions remove information, whereas prose is a lossless serialization
control. The histories are constructed text records, not visual or physical
counterfactual rollouts.

## Recommended example caption: formal_000

**Same world, different acquired information.** In the constructed vase example,
the world location is bench_18 in every variant. Agent_1 retains bed_11 when the
move is unseen and no relevant report arrives, including when a relevant report
is not delivered; it reports bench_18 after witnessing the change or receiving
the report. Agent_0 retains bed_11 throughout. Both linked SSC and lossless prose
produce all four correct state/citation outputs. All shown answers are saved
outputs; the example illustrates information-sensitive reasoning, not a unique
advantage of code syntax.

## Optional favorable and reverse examples

In formal_007, with an undelivered relevant report, SSC preserves the target's
earlier supported location table_30, whereas prose outputs the unobserved
intermediate location couch_61. In formal_019 the direction reverses: SSC makes
state errors in the witnessed/unseen variants while prose answers all four
histories correctly. Both are post-hoc qualitative selections; neither is used
to estimate aggregate gains. No claim is made about an internal forgetting
mechanism or about causal internal reasoning from these output differences.

## Disclosure language

The protocol and sample split were frozen locally before formal inference; this
was not an external preregistration. During engineering, semantic citation-ID
errors were separated from whole-response syntax failures. The 88 already obtained
responses were reused byte-for-byte rather than resampled. All source snapshots,
formal responses, rejected outputs, and deterministic evaluation code are released.
