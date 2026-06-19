# AIE Fine-Grid Simulation Design

## Objective

Create `AIE_re2.1.py` from `AIE_re2.py` so a fixed 300 x 300 projected grayscale mask is optimized using a 600 x 600 numerical physics grid. The refinement improves numerical sampling without claiming additional independently controllable projector pixels.

## Physical and numerical grids

- The projected mask remains 300 x 300 with a physical pixel pitch of 7.395 um.
- The simulation and curing target use a 600 x 600 grid with a cell pitch of 3.6975 um.
- The physical field of view remains 2.2185 mm x 2.2185 mm.
- Each projected pixel is represented by a constant 2 x 2 block of fine simulation cells using differentiable nearest-neighbor expansion (`repeat_interleave`).
- Scattering is applied after this expansion. Diffusion, dose accumulation, degree of conversion, and loss evaluation all operate on the fine grid.

## Optimization data flow

1. Load and normalize a 600 x 600 target image.
2. Create a 300 x 300 initial projector mask by 2 x 2 area averaging the target.
3. Store only this coarse mask as the trainable `torch.nn.Parameter`.
4. Expand the coarse mask to 600 x 600 before applying the optical and reaction model.
5. Compare the simulated 600 x 600 DoC fields with the native 600 x 600 target.
6. Backpropagate the fine-grid loss through the expansion. PyTorch sums the four fine-cell gradient contributions into each corresponding projector-pixel parameter.
7. Save the optimized projector mask at 300 x 300. Fine-grid blurred intensity and DoC outputs remain 600 x 600 diagnostic results.

## Input validation

- The target must be a two-dimensional grayscale representation after image loading.
- Both target dimensions must be exactly twice the fixed projector dimensions; the initial implementation therefore requires 600 x 600.
- Invalid dimensions fail early with an explicit error rather than silently resizing or changing the modeled field of view.

## Implementation scope

- Preserve `AIE_re2.py` unchanged because it contains user modifications.
- Add the new implementation as `AIE_re2.1.py`, as requested.
- Keep existing material parameters, time integration, loss definition, output generation, and optimizer settings unless a grid separation requires a dimensional correction.
- Name coarse projector dimensions separately from fine simulation dimensions to prevent accidental reuse.
- Recompute all pixel-unit diffusion and scattering kernels from the fine cell pitch.

## Verification

- A focused automated test will verify that a 300 x 300 projector mask expands to 600 x 600 as constant 2 x 2 blocks and remains differentiable.
- The test will verify that gradients from all four fine cells accumulate into their parent projector pixel.
- Static checks will verify that the trainable parameter and saved projector mask remain 300 x 300 while the physics arrays and target are 600 x 600.
- A lightweight smoke path will avoid running the full 1000-epoch, 250-step production optimization during validation.

## Limitations

The finer grid improves discretization of diffusion, scattering, and curing boundaries. It does not make the projector independently control 600 x 600 pixels. Subpixel target features that cannot be produced by the coarse projected field may retain nonzero optimization error.
