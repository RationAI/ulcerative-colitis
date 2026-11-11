# TODO

Open work on the concept explainability of the Nancy model (branch `feature/explainability`).
Last updated 2026-11-11.

## Data: white holes in the CZI pyramids

Some CZI slides have white rectangles over real tissue at levels 1 to 3, while other levels show
the tissue. OpenSlide and wsi-service return the same pixels. This hurts the models in two ways:
tissue tiled at a level with a hole reaches the model as white, and tissue under a level-3 hole is
left out of the tissue mask (made at level 3) and so never tiled.

Scanned on 2026-11-10 by `white_holes.py` (job `uc-white-holes`): every Nancy slide at level 2 and
every tiled dysplasia slide at levels 1 and 2, compared with the other levels 1 to 4. Results in
`/mnt/projects/inflammatory_bowel_disease/ulcerative_colitis/white_holes/` (`holes.parquet`,
`mask_missed_by_hole.parquet`, per-slide `grids/` and `previews/`). Level 0 was not read, so a
hole in all of levels 1 to 4 would be missed. Level 3 alone would have missed half the Nancy hole
tissue.

| | Slides | Tissue hole of 100+ cells | Lose 1%+ / 5%+ of tissue | Tissue lost | Tiles half over a hole | Mask missed tissue to a level-3 hole |
|---|---|---|---|---|---|---|
| FTN | 1,339 | 169 | 87 / 14 | 0.32% | 1,039 | 0.22% |
| IKEM (incl. validation) | 2,316 | 224 | 82 / 8 | 0.14% | 852 | 0.10% |
| KNL_PATOS | 176 | 21 | 8 / 2 | 0.83% | 433 | 0.81% |
| Nancy, all | 3,831 | 414 | 177 / 24 | 0.23% | 2,324 of 2.24 M | 0.18% |
| Dysplasia L2 / L1 | 229 | 32 / 15 | 9 / 0 at L2 | 0.23% / 0.08% | 112 / 133 | 0.18% |

A cell is 81 level-0 px, about 14 um. `mask_missed_fraction` in `holes.parquet` also counts pale
pink mucus and mounting medium that the mask rightly excluded; use `mask_missed_by_hole.parquet`
(missed tissue where level 3 is blank) for the hole part. Worst slides: 5503_25_A_HE11 (13% lost,
11% never tiled), 20673_25 and 4262_21 (about 20% never tiled), 3572_24, 43205_24, 5716_24_HE,
18749_24 (10 to 12% lost).

- [x] Scan all slides in all splits as a job.
- [x] Read affected slides with a second CZI reader (pylibCZIrw and aicspylibczi, 10 spots).
      The files are fine. The holes sit where the rectangles of two scenes overlap: one scene
      stores tissue there, the other white glass, at every level. The reader paints one over the
      other, and OpenSlide puts the white scene on top at some levels. libCZI got the tissue in 9
      of 10 spots but the white scene in the 10th (5201_25_A_HE01), so neither reader is safe.
- [x] Patch the OpenSlide fork in the images (iewchen `zeiss-czi-jxr-simd-gray`). It draws with
      SATURATE, so the first tile drawn wins whatever scene it is from. The patch makes the pure
      white padding transparent and fills white behind the tiles at the end. Tested on the 10
      spots: holes filled at levels 1 and 2, only padding and its seam pixels change, same speed.
- [x] Push the patch to RationAI/openslide, branch `czi-scene-overlap`, commit `5db4e1c`.
- [x] Get the images rebuilt against `5db4e1c` (`cerit.io/rationai/base:cuda-3.0.1`) and rerun
      `white_holes.py` (job `uc-white-holes-patched`, results in `white_holes_patched/`). All
      4,289 slide-levels: 0 hole cells, 0 tiles hit, 0 errors (before: 999,179 hole tissue cells,
      2,569 tiles hit). Masks and tiling there are still the old ones, so mask-missed is unchanged.
- [ ] Delete `white_holes.py` once nothing else needs it.
- [ ] Decide whether to re-tile and re-embed the affected slides with it.
- [ ] Decide what to do. Overall loss is under 1%, but about 25 slides lose 5% or more, and
      KNL_PATOS (the cross-centre test) is hit hardest.
- [ ] Find out what the two slides that are white at every level (7198_22_HE_01, 646_21) have in
      common.
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

- [x] Run all concepts for both K. `pathologist.html` is about 24 MB at K = 24 and 12 MB at K = 16,
      because each link carries the row's session. If that's slow, build the links in JavaScript
      on click.
- [x] Rerun both K with overlay opacity 0.5, so the links in `pathologist.html` pick it up.
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
