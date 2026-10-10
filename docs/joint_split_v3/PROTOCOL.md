# Joint Dataset Split v3

This document records the user-confirmed target protocol for the PhysEdit dataset benchmark. It is a design specification; target counts do not imply that the media inventory, quality review, or frozen split manifests are complete.

## Training and evaluation protocol

Train jointly on Genesis and Isaac data. Use 16 events for training, validation, and in-distribution (ID) testing, with an 80% / 10% / 10% split by source groups. Keep four entire event classes out of training:

| Partition | Events | Purpose |
|---|---|---|
| Train pool | T01, T02, T04–T10, T12–T18 | Joint train / validation / ID test, split by groups |
| OOD-1 | T03, T20 | Transfer to unseen cascade events |
| OOD-2 | T11, T19 | Transfer to unseen relay and multi-object interaction events |

The OOD sets are whole-event transfer evaluations. OOD-1 is not claimed to be a pure short-chain-to-long-chain test; OOD-2 does not isolate a single changed factor. Neither OOD set is an engine holdout because training uses both engines.

## Target capacity

The design target is 52,800 Genesis videos and 20,160 Isaac videos (72,960 total). At target capacity, the 16-event train pool contains 58,800 videos, split into 47,040 train, 5,880 validation, and 5,880 ID test. OOD-1 targets 6,240 videos and OOD-2 targets 7,920 videos. Isaac target counts are zero for events without planned Isaac data; missing media or cross-engine pairs must not be fabricated.

These are capacity targets, not a statement of current usable inventory. Final counts may change after complete media, state, metadata, and quality audits, and whole-group integrity takes priority over exact ratios.

## Grouping and release requirements

Keep all views of a trajectory, background variants of one physical configuration, branches and repeats from one critical family, known cross-engine counterparts, aliases/duplicate media, and contiguous scan intervals together. Review numeric intervals and boundary guards before assigning continuous scans (especially T06, T09, and T10). Where feasible, retain all four backgrounds and both global and critical samples across train, validation, and ID test.

Before release, pin the actual media and state versions; audit views, metadata, states, and runtime parameters; validate event labels and visual physical quality; resolve cross-engine configuration mappings; separately audit metric-development exposure; then assign whole groups, verify coverage, and freeze manifests.


