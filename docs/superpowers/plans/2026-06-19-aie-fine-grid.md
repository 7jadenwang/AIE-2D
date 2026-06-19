# AIE Fine-Grid Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `AIE_re2.1.py`, which optimizes a fixed 300 x 300 projector mask against a 600 x 600 AIE physics simulation and target.

**Architecture:** Put the small, testable grid transformations in `aie_fine_grid.py`. The new simulation script copies the existing AIE workflow, initializes the coarse projector mask by 2 x 2 area averaging, expands that trainable mask into constant 2 x 2 simulation blocks before scattering, and keeps all reaction physics and loss evaluation on the fine grid.

**Tech Stack:** Python, PyTorch, NumPy, OpenCV, Pillow, pytest

---

### Task 1: Fine-grid transformation functions

**Files:**
- Create: `aie_fine_grid.py`
- Create: `tests/test_aie_fine_grid.py`

- [ ] **Step 1: Write failing shape, averaging, and gradient tests**

```python
import pytest
import torch

from aie_fine_grid import expand_projector_mask, initialize_projector_mask


def test_initialize_projector_mask_averages_each_fine_block():
    target = torch.tensor([[0.0, 2.0], [4.0, 6.0]])
    result = initialize_projector_mask(target, (1, 1), refinement=2)
    torch.testing.assert_close(result, torch.tensor([[3.0]]))


def test_expand_projector_mask_creates_constant_blocks_and_accumulates_gradient():
    mask = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    fine = expand_projector_mask(mask, refinement=2)
    assert fine.shape == (4, 4)
    torch.testing.assert_close(fine[:2, :2], torch.ones(2, 2))
    fine.sum().backward()
    torch.testing.assert_close(mask.grad, torch.full((2, 2), 4.0))


def test_production_grid_round_trip_has_expected_dimensions():
    target = torch.zeros(600, 600)
    coarse = initialize_projector_mask(target, (300, 300), refinement=2)
    fine = expand_projector_mask(coarse, refinement=2)
    assert coarse.shape == (300, 300)
    assert fine.shape == target.shape


def test_initialize_projector_mask_rejects_wrong_target_shape():
    with pytest.raises(ValueError, match="expected target shape"):
        initialize_projector_mask(torch.zeros(599, 600), (300, 300), refinement=2)
```

- [ ] **Step 2: Run the tests and verify they fail because the module is absent**

Run: `python -m pytest tests/test_aie_fine_grid.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'aie_fine_grid'`.

- [ ] **Step 3: Implement the minimal grid functions**

```python
import torch
import torch.nn.functional as F


def _validate_2d(tensor, name):
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(tensor.shape)}")


def initialize_projector_mask(target, projector_shape=(300, 300), refinement=2):
    _validate_2d(target, "target")
    expected = tuple(size * refinement for size in projector_shape)
    if tuple(target.shape) != expected:
        raise ValueError(f"expected target shape {expected}, got {tuple(target.shape)}")
    return F.avg_pool2d(target[None, None], refinement, refinement)[0, 0]


def expand_projector_mask(mask, refinement=2):
    _validate_2d(mask, "projector mask")
    if refinement < 1:
        raise ValueError("refinement must be at least 1")
    return mask.repeat_interleave(refinement, 0).repeat_interleave(refinement, 1)
```

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_aie_fine_grid.py -v`

Expected: four tests pass.

- [ ] **Step 5: Commit the transformation functions and tests**

```powershell
git add -- aie_fine_grid.py tests/test_aie_fine_grid.py
git commit -m "test: define AIE fine-grid transformations"
```

### Task 2: New fine-grid AIE optimization script

**Files:**
- Create: `AIE_re2.1.py`
- Modify: `tests/test_aie_fine_grid.py`

- [ ] **Step 1: Add a failing source-contract test**

```python
from pathlib import Path


def test_fine_grid_script_preserves_projector_grid_and_expands_before_physics():
    source = Path("AIE_re2.1.py").read_text(encoding="utf-8")
    assert "PROJECTOR_SHAPE = (300, 300)" in source
    assert "REFINEMENT = 2" in source
    assert "PROJECTOR_PITCH = 7.395e-6" in source
    assert "dx = dy = PROJECTOR_PITCH / REFINEMENT" in source
    assert "initialize_projector_mask(target_mask, PROJECTOR_SHAPE, REFINEMENT)" in source
    assert "expand_projector_mask(opt_mask, REFINEMENT)" in source
```

- [ ] **Step 2: Run the contract test and verify it fails because the script is absent**

Run: `python -m pytest tests/test_aie_fine_grid.py::test_fine_grid_script_preserves_projector_grid_and_expands_before_physics -v`

Expected: failure with `FileNotFoundError` for `AIE_re2.1.py`.

- [ ] **Step 3: Copy the current simulation and separate its grids**

Create `AIE_re2.1.py` from the current `AIE_re2.py`, preserving the original file. Add:

```python
from aie_fine_grid import expand_projector_mask, initialize_projector_mask

PROJECTOR_SHAPE = (300, 300)
REFINEMENT = 2
PROJECTOR_PITCH = 7.395e-6
dx = dy = PROJECTOR_PITCH / REFINEMENT
```

Load `Lshape600.png`, normalize it as the native fine target, and initialize only the coarse trainable parameter:

```python
img = Image.open("./Lshape600.png")
# Keep the existing mode-aware normalization logic.
H, W = target.shape
target_mask = torch.tensor(target * 255, dtype=torch.float32, device=device)
initial_mask = initialize_projector_mask(target_mask, PROJECTOR_SHAPE, REFINEMENT)
opt_mask = torch.nn.Parameter(initial_mask.clone())
```

At the beginning of every epoch, expand the projector field before optical scattering:

```python
fine_mask = expand_projector_mask(opt_mask, REFINEMENT)
opt_mask_pre = fine_mask.unsqueeze(0).unsqueeze(0)
```

Use `target_mask / 255` for all fine-grid loss targets. At final export, save `opt_mask` directly as the 300 x 300 projector image, but expand it again before calculating and saving the 600 x 600 final blurred field:

```python
final_opt_mask = opt_mask.detach().cpu().numpy() / 255 * 65535
fine_mask = expand_projector_mask(opt_mask, REFINEMENT)
blur_mask_pre = fine_mask.unsqueeze(0).unsqueeze(0)
```

- [ ] **Step 4: Run focused tests and compile the new script**

Run: `python -m pytest tests/test_aie_fine_grid.py -v`

Expected: all tests pass.

Run: `python -m py_compile AIE_re2.1.py aie_fine_grid.py`

Expected: exit code 0 with no output.

- [ ] **Step 5: Run a static dimensional audit**

Run: `rg -n "PROJECTOR_SHAPE|REFINEMENT|PROJECTOR_PITCH|dx =|target_mask|initial_mask|fine_mask|final_opt_mask" AIE_re2.1.py`

Expected: the trainable and saved projector mask remain coarse; `fine_mask`, the physics arrays, and the target use the fine dimensions; `dx` is derived from physical projector pitch.

- [ ] **Step 6: Commit the fine-grid script**

```powershell
git add -- AIE_re2.1.py tests/test_aie_fine_grid.py
git commit -m "feat: add fine-grid AIE optimization"
```

### Task 3: Verification and regression review

**Files:**
- Review: `AIE_re2.1.py`
- Review: `aie_fine_grid.py`
- Review: `tests/test_aie_fine_grid.py`

- [ ] **Step 1: Run all focused automated verification**

Run: `python -m pytest tests/test_aie_fine_grid.py -v`

Expected: all tests pass.

Run: `python -m py_compile AIE_re2.1.py aie_fine_grid.py`

Expected: exit code 0.

- [ ] **Step 2: Review the final diff for unintended changes**

Run: `git diff --check HEAD~2..HEAD`

Expected: no whitespace errors.

Run: `git status --short`

Expected: pre-existing user modifications to `AIE_re2.py` and the notebook remain untouched; no implementation files are left uncommitted.

- [ ] **Step 3: Confirm requirement evidence**

Confirm from tests and source that the target and physics grid are 600 x 600, the only trainable projector field is 300 x 300, each projected pixel becomes a constant 2 x 2 fine block, fine-grid gradients reach the coarse parameter, and the physical field of view remains unchanged.
