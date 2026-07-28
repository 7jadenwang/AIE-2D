# Shape-Agnostic Sharp-Corner Loss

## Goal

Add a clean `intensityOptLoss_v2` to `AIE_TEMPOv1.1.py` that preserves the
existing three-time-point intensity objective while giving sharp target
corners more influence over the optimized final degree-of-conversion (DoC)
field. The existing `intensityOptLoss` remains unchanged.

## Design

The new loss will generate a fixed corner-emphasis map from the target using
PyTorch operations:

1. Compute horizontal and vertical target gradients with Sobel kernels.
2. Form and locally average the structure-tensor terms.
3. Compute a Harris response and keep only positive responses.
4. Normalize each image's response safely and expand it over a small
   neighborhood with max pooling.

This detects locations where target gradients change strongly in both
directions, emphasizing sharp corners without applying the same weight along
all straight edges. Because the map comes from the target rather than
hard-coded coordinates, the loss applies to multiple target shapes.

The objective is:

`total = base_intensity_loss + corner_weight * corner_loss`

The base term uses the same first, intermediate, and final DoC targets as the
current loss: `0.0`, `0.77`, and `0.91` times the target. The corner term is
the weighted root-mean-square final-DoC residual relative to `0.91 * target`.
It is normalized by the sum of corner weights so its scale does not grow with
image size or corner count.

## Interface

`intensityOptLoss_v2(firstDoC, intermediateDoC, finalDoC, target, ...)`

Optional parameters will expose only the meaningful tuning controls:

- `corner_weight`: strength of the added corner term.
- `corner_window`: odd structure-tensor averaging window.
- `corner_radius`: neighborhood radius around detected corners.
- `harris_k`: Harris detector sensitivity.
- `eps`: numerical safeguard.

Inputs may be 2-D image tensors, matching the current optimizer. Device and
dtype follow `finalDoC`; the target is converted consistently.

## Integration

The optimization site will retain the existing loss call as a comment and use
`intensityOptLoss_v2` in one explicit line. This makes rollback and comparison
straightforward. No physics, target loading, optimizer, or output code changes
are in scope.

## Validation

Focused tests or an isolated validation script will check:

- Finite scalar loss and finite gradients.
- A rounded/errorful sharp corner receives a larger penalty than the same
  residual on a straight edge.
- A target without corners produces a finite result and falls back cleanly to
  the base loss behavior.
- Existing `intensityOptLoss` output is unchanged.
- CPU/GPU device and float dtype handling remain consistent.

The full physical optimization is not required for unit validation because it
is expensive; a subsequent L-shape optimization run is the empirical
acceptance test.

## Success Criteria

- Sharp target corners automatically receive additional optimization weight.
- Straight edges and smooth regions are not globally over-weighted.
- The loss remains differentiable with respect to all DoC inputs used by the
  base objective and to `finalDoC` in the corner term.
- Existing behavior remains available through `intensityOptLoss`.
