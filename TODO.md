# TODO

Open work on the concept explainability of the Nancy model (branch `feature/explainability`).
Last updated 2026-11-06.

## Data: white holes at level 2

Some CZI slides have white rectangles over real tissue at level 2, while level 0 shows the
tissue and level 1 has smaller holes. Tiles and embeddings are read at level 2, so the model saw
white in those places. OpenSlide and wsi-service return the same pixels.

| Slide | White at level 2 | Level 1 | Level 0 |
|---|---|---|---|
| 9656_24_HE | 95% | 7% | 0% |
| 2256_23_HE | 97% | 16% | 0% |
| 46382_24 | 65% | 7% | 0% |

These are three of five blank gallery tiles checked in `test_preliminary`, picked because they
were blank, so they say nothing about how common this is. Two others (7198_22_HE_01, 646_21) are
white at every level, so they have a different cause.

- [ ] **Scan all slides in all splits as a job:** compare white at level 2 with level 0 on
      low-resolution versions, and report the affected slides and the tissue lost.
- [ ] **Read one affected slide with a second CZI reader** to find out whether the files' lower
      levels are incomplete or OpenSlide's Zeiss reader mishandles them.
- [ ] Decide what to do from those two results. If few slides are affected, document it. If many
      are, regenerate level 2, redo the embeddings and consider retraining the heads.
- [ ] Find out what the two slides that are white at every level have in common.
- [ ] List the slides behind the blank top tiles in the K = 16 and K = 24 galleries.

Do not filter blank tiles out of the galleries. They are how this was found.

## Concepts: choose K and name them

Candidates: flat semi-NMF under the output metric, ridge 0.3, seed 42.

| | K = 16 | K = 24 |
|---|---|---|
| Fit run | `be859ae30ca143b78d957493d4f8a8c8` | `5133f515b57641edb7ba0bee7b4aade9` |
| Exact agreement, 3-seed mean | .803 | .876 |
| Active concepts (full run) | 11 | 14 |
| Minor concepts (under 1.5 points) | 5 | 10 |

- [ ] Review `pathologist.html` for both K in a browser. It hasn't been looked at yet.
- [ ] Choose K. K = 24 is the current recommendation, but it needs the review above.
- [ ] Decide whether the "Minor concepts" section stays during blind naming. It reveals that those
      concepts matter less, though not for which grade.
- [ ] Decide whether the page needs a name and comment field with CSV export, as `tree.html` has.
- [ ] Have the pathologist name the concepts blind, with numbers hidden. Provisional names from
      other dictionaries must not reach them.
- [ ] Check the chosen K's concepts on other seeds. Galleries exist for seed 42 only.

## Concept masks in xOpat

Clicking a gallery tile opens the slide with that concept's mask visible and the other concepts'
masks loaded but hidden. The pathologist uses the masks to name concepts, so both K need them.

Agreed design: one heatmap per concept, value = share `v_k / sum v`, colour scale global per
concept, gallery slides only (about 158 slides at K = 24, 109 at K = 16), and low-code tiles
treated like any other tile. Masks are limited to tile resolution: one value every 112 level-2 px,
about 175 µm. Write them with `ratiopath`, not `rationai-masks`, which is being deprecated.

- [ ] Run all concepts for both K. `pathologist.html` will grow to about 17 MB at K = 24, because
      each link carries the row's session, unless links are built in JavaScript on click.
- [ ] Mark level 2 white holes on the masks. xOpat can show level 0 tissue there, which the model
      never saw.

### Moving between the slides of a concept row

One session per concept row works: the pilot showed the overlays, all 8 slides in the Slide
Switcher, and switching between them with each slide's own masks.

- [ ] After a switch the view doesn't centre on the slide's tile, because `params.viewport` is per
      viewer, not per slide. Ask the xOpat maintainers for a per-background viewport.
- [ ] Ask the xOpat maintainers for on-screen buttons. `_createControlButtons` in
      `plugins/slide-info/slide-info.mjs` has them commented out, with a todo about multiple viewers.

Embedding xOpat in our own page with buttons isn't possible from MLflow: it sends
`X-Frame-Options: SAMEORIGIN`.

## Reporting

Everything so far was chosen on `test_preliminary`. `test_final` only reports and never selects.

- [ ] Run `concept_completeness` and `concept_gallery` for the chosen dictionary on `test_final`.
- [ ] Run the same on `knl_patos` for cross-centre transfer. Earlier dictionaries showed attention
      not transferring, so this needs repeating on the metric fit.
- [ ] Add a dictionary config and experiment for the chosen fit. Jobs currently use the `flat_k12`
      experiment with a `dictionary_uri` override.

## Open questions

- [ ] No K gives NHI 2 its own concept. Decide whether to accept this or look further.
- [ ] A shared factor still moves all codes together (PC1 .48 at K = 16, .74 at K = 24). Share
      ranking handles it in the gallery, but it needs a sentence in the method description.
- [ ] Decide how to present NHI 4 effects, which stay small (at most 2.1 points) because those
      slides are near 99% confidence.
- [ ] Check whether any concept in the chosen dictionary attracts attention on blur or artefacts,
      as concept 10 of the old dictionary did.
