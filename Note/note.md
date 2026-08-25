# Light Scattering vs. Optimization Conditioning

Notes from comparing `AIE_TEMPOv1.1.py` optimization with and without the light-scattering
convolution (`blur_size` / the `ls` kernel, originally set via lines ~223-230, applied at line 244).

## What the scattering term does

`blur_mask = conv2d(opt_mask, ls)` — a Gaussian blur (kernel derived from `blur_size`) applied to
`opt_mask` every epoch before it drives the cure physics (`energy`, `B`, dose accumulation, DoC).
Setting `blur_size=0` collapses `ls` to a trivial 1x1 kernel, i.e. `blur_mask == opt_mask`
(no scattering) — functionally the same as the old manual override at line 245.

## Observation

With scattering on (`blur_size=50e-6`), the final training loss is much lower than with
`blur_size=0`, after the same 500 epochs / lr=0.77.

## Why (optimization dynamics, not representational capacity)

Removing the blur *increases* `opt_mask`'s degrees of freedom (fully independent per-pixel
control), so it should in principle be at least as expressive. The gap is instead caused by how
well Adam can descend the loss landscape in a fixed number of steps:

- The cure physics has several hard nonlinearities per timestep: `clamp(min=0)` on O2/TEMPO,
  `torch.where(O2<=0 & TEMPO<=0, ...)` (a genuine gradient discontinuity at the threshold-crossing
  step), `clamp(min=1e-12)` denominators, and the saturating `1-exp(-B*t)` cure curve.
- Without blur, each `opt_mask` pixel is optimized almost independently through this rough
  landscape (only coupled via O2/TEMPO diffusion). Pixels near sharp target edges (stairs target)
  tend to rocket to the `opt_mask.data.clamp_(0,255)` boundary and get stuck with near-zero local
  gradient, while neighbors don't coordinate.
- Because the Gaussian blur kernel is symmetric, backprop through it is *also* a Gaussian blur:
  every `opt_mask` pixel's gradient becomes an average over its neighborhood rather than one
  noisy per-pixel signal. This is the same "blurred gradient" trick used in pixel-space image
  optimization (feature visualization / DeepDream) to avoid high-frequency noise and saturation
  traps.
- Net effect: scattering acts as an implicit gradient preconditioner. It's not that the model is
  more powerful with scattering — it's that Adam converges much better within a fixed epoch budget.

## Fix: precondition the gradient directly, independent of physical scattering

Added a **separate** Gaussian kernel (`grad_smooth_kernel`, `grad_smooth_sigma=2.0` px) decoupled
from `blur_size`, so it stays non-trivial even when the physical scattering kernel is a 1x1
identity. Applied only when `blur_size==0`:

```python
Loss.backward()
if blur_size==0:
    with torch.no_grad():
        grad_padded = F.pad(opt_mask.grad.view(1,1,H,W),
                             pad=(grad_smooth_pad,)*4, mode='reflect')
        opt_mask.grad = F.conv2d(grad_padded, grad_smooth_kernel)[0,0]
optimizer.step()
```

This keeps the forward physics as the true zero-scattering model (what actually gets simulated
and compared to `target`) while still giving Adam a smooth, well-conditioned gradient signal to
descend — separating "is scattering physically real" from "does the optimizer need smoothing to
converge."

## Open items / things to try if the gap persists

- Lower `lr` for the `blur_size==0` case (0.77 was likely tuned assuming blur-smoothed gradients).
- Anneal `grad_smooth_sigma` toward 0 over epochs (coarse-to-fine / graduated non-convexity) for a
  sharper final result once the optimizer is out of the bad early basin.
- Compare `aaa_loss_history.png` curves across runs — a noisy/plateaued curve without smoothing
  vs. a smooth descent with it would confirm the conditioning explanation.
