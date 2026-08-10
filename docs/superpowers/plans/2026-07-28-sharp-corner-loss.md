# Sharp-Corner Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and activate a shape-agnostic `intensityOptLoss_v2` that increases the optimization penalty near sharp target corners while preserving the existing intensity objective.

**Architecture:** Keep the original loss unchanged in `AIE_TEMPOv1.1.py`. Add one focused helper that produces a normalized Harris corner-neighborhood map from the fixed target, then add a v2 loss that combines the original loss with a normalized, corner-weighted final-DoC RMS residual. Test the functions in isolation because importing the simulation script directly starts the full experiment.

**Tech Stack:** Python, PyTorch, pytest, Python `ast`

## Global Constraints

- Modify only `AIE_TEMPOv1.1.py` and add one focused test file.
- Keep `intensityOptLoss` unchanged.
- Use only PyTorch operations for corner detection and loss computation.
- Do not change the simulation physics, target loading, optimizer, or output code.
- Preserve device and floating-point dtype consistency with `finalDoC`.
- Keep each new Python function within 40 lines where reasonably possible.

---

### Task 1: Harris Corner Weight Map

**Files:**
- Modify: `AIE_TEMPOv1.1.py:35`
- Create: `tests/test_corner_loss.py`

**Interfaces:**
- Consumes: 2-D `target: torch.Tensor`, odd positive `corner_window: int`, nonnegative `corner_radius: int`, `harris_k: float`, and `eps: float`.
- Produces: `_harris_corner_weights(target, corner_window=5, corner_radius=3, harris_k=0.04, eps=1e-8) -> torch.Tensor`, a detached 2-D map in `[0, 1]` with the target's device and dtype.

- [ ] **Step 1: Write the isolated function loader and failing corner-map tests**

Create `tests/test_corner_loss.py` with a loader that parses `AIE_TEMPOv1.1.py`, compiles only `intensityOptLoss`, `_harris_corner_weights`, and `intensityOptLoss_v2`, and executes them with `torch` and `torch.nn.functional as F` supplied in the namespace. This avoids running the top-level physical simulation.

```python
import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


SCRIPT = Path(__file__).parents[1] / "AIE_TEMPOv1.1.py"


def load_loss_functions():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = {"intensityOptLoss", "_harris_corner_weights", "intensityOptLoss_v2"}
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace


def make_l_target(size=32):
    target = torch.zeros(size, size)
    target[8:24, 8:14] = 1
    target[18:24, 8:24] = 1
    return target


def test_harris_corner_weights_are_finite_normalized_and_shape_preserving():
    fn = load_loss_functions()["_harris_corner_weights"]
    target = make_l_target().to(torch.float64)
    weights = fn(target)
    assert weights.shape == target.shape
    assert weights.dtype == target.dtype
    assert torch.isfinite(weights).all()
    assert 0 <= weights.min() <= weights.max() <= 1
    assert weights.max() == pytest.approx(1.0)


def test_harris_corner_weights_reject_invalid_window():
    fn = load_loss_functions()["_harris_corner_weights"]
    with pytest.raises(ValueError, match="odd positive"):
        fn(make_l_target(), corner_window=4)
```

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```powershell
pytest tests/test_corner_loss.py -v
```

Expected: FAIL because `_harris_corner_weights` and `intensityOptLoss_v2` are not yet defined in `AIE_TEMPOv1.1.py`.

- [ ] **Step 3: Implement the minimal Harris helper**

Add below `intensityOptLoss`:

```python
def _harris_corner_weights(
    target, corner_window=5, corner_radius=3, harris_k=0.04, eps=1e-8
):
    if corner_window < 1 or corner_window % 2 == 0:
        raise ValueError('corner_window must be an odd positive integer')
    if corner_radius < 0:
        raise ValueError('corner_radius must be nonnegative')

    image = target.detach().unsqueeze(0).unsqueeze(0)
    sobel_x = image.new_tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ).view(1, 1, 3, 3) / 8
    gx = F.conv2d(image, sobel_x, padding=1)
    gy = F.conv2d(image, sobel_x.transpose(-1, -2), padding=1)
    pad = corner_window // 2
    tensor_terms = [
        F.avg_pool2d(term, corner_window, stride=1, padding=pad)
        for term in (gx * gx, gy * gy, gx * gy)
    ]
    sxx, syy, sxy = tensor_terms
    response = (sxx * syy - sxy.square()) - harris_k * (sxx + syy).square()
    response = response.clamp_min(0)
    response = response / response.amax().clamp_min(eps)
    if corner_radius:
        size = 2 * corner_radius + 1
        response = F.max_pool2d(response, size, stride=1, padding=corner_radius)
    return response[0, 0]
```

- [ ] **Step 4: Run the corner-map tests**

Run:

```powershell
pytest tests/test_corner_loss.py -v
```

Expected: the corner-map tests PASS; later v2 tests are not present yet.

- [ ] **Step 5: Commit the independently tested helper**

```powershell
git add AIE_TEMPOv1.1.py tests/test_corner_loss.py
git commit -m "feat: add target-derived corner weights"
```

### Task 2: Corner-Weighted Intensity Loss

**Files:**
- Modify: `AIE_TEMPOv1.1.py:35-65`
- Modify: `tests/test_corner_loss.py`

**Interfaces:**
- Consumes: `firstDoC`, `intermediateDoC`, `finalDoC`, and `target` as same-shaped 2-D tensors plus optional `corner_weight=1.0`, `corner_window=5`, `corner_radius=3`, `harris_k=0.04`, and `eps=1e-8`.
- Produces: `intensityOptLoss_v2(...) -> torch.Tensor`, a finite scalar differentiable with respect to the three DoC tensors.
- Uses: `_harris_corner_weights(...) -> torch.Tensor` from Task 1.

- [ ] **Step 1: Add failing behavioral and gradient tests**

Append:

```python
def test_v2_equals_base_when_corner_weight_is_zero():
    funcs = load_loss_functions()
    target = make_l_target()
    first = torch.zeros_like(target)
    intermediate = 0.77 * target
    final = 0.91 * target
    base = funcs["intensityOptLoss"](first, intermediate, final, target)
    actual = funcs["intensityOptLoss_v2"](
        first, intermediate, final, target, corner_weight=0
    )
    assert torch.allclose(actual, base)


def test_v2_penalizes_corner_error_more_than_straight_edge_error():
    funcs = load_loss_functions()
    target = make_l_target()
    weights = funcs["_harris_corner_weights"](target)
    corner_index = tuple(torch.nonzero(weights == weights.max())[0].tolist())
    edge_candidates = torch.nonzero((weights == 0) & (target == 1))
    edge_index = tuple(edge_candidates[len(edge_candidates) // 2].tolist())

    expected = 0.91 * target
    first = torch.zeros_like(target)
    intermediate = 0.77 * target
    corner_error = expected.clone()
    edge_error = expected.clone()
    corner_error[corner_index] -= 0.2
    edge_error[edge_index] -= 0.2

    corner_loss = funcs["intensityOptLoss_v2"](
        first, intermediate, corner_error, target
    )
    edge_loss = funcs["intensityOptLoss_v2"](
        first, intermediate, edge_error, target
    )
    assert corner_loss > edge_loss


def test_v2_has_finite_gradients_and_flat_target_fallback():
    fn = load_loss_functions()["intensityOptLoss_v2"]
    target = torch.zeros(24, 24, dtype=torch.float64)
    docs = [
        torch.zeros_like(target, requires_grad=True),
        torch.zeros_like(target, requires_grad=True),
        torch.full_like(target, 0.1, requires_grad=True),
    ]
    loss = fn(*docs, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(doc.grad is not None for doc in docs)
    assert all(torch.isfinite(doc.grad).all() for doc in docs)
```

- [ ] **Step 2: Run the v2 tests and verify failure**

Run:

```powershell
pytest tests/test_corner_loss.py -v
```

Expected: FAIL because `intensityOptLoss_v2` is not defined.

- [ ] **Step 3: Implement the minimal v2 loss**

Add below `_harris_corner_weights`:

```python
def intensityOptLoss_v2(
    firstDoC, intermediateDoC, finalDoC, target, corner_weight=1.0,
    corner_window=5, corner_radius=3, harris_k=0.04, eps=1e-8
):
    target = target.to(device=finalDoC.device, dtype=finalDoC.dtype)
    base_loss = intensityOptLoss(firstDoC, intermediateDoC, finalDoC, target)
    weights = _harris_corner_weights(
        target, corner_window, corner_radius, harris_k, eps
    )
    residual = finalDoC - 0.91 * target
    weighted_mse = (weights * residual.square()).sum()
    corner_loss = torch.sqrt(weighted_mse / weights.sum().clamp_min(eps) + eps)
    has_corners = (weights.sum() > eps).to(finalDoC.dtype)
    return base_loss + corner_weight * has_corners * corner_loss
```

The `has_corners` multiplier ensures a flat or corner-free target returns the
base loss exactly instead of adding `sqrt(eps)`.

- [ ] **Step 4: Run the focused test suite**

Run:

```powershell
pytest tests/test_corner_loss.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit the v2 loss**

```powershell
git add AIE_TEMPOv1.1.py tests/test_corner_loss.py
git commit -m "feat: add corner-weighted intensity loss"
```

### Task 3: Activate v2 and Review the Integration

**Files:**
- Modify: `AIE_TEMPOv1.1.py:298-302`
- Test: `tests/test_corner_loss.py`

**Interfaces:**
- Consumes: existing simulation tensors `DoC[tstepT0]`, `DoC[tstepT1]`, `DoC[tstepT2]`, and `mask / 255`.
- Produces: active scalar `Loss` from `intensityOptLoss_v2`; retains the original call as the immediate commented comparison path.

- [ ] **Step 1: Add a failing source-level activation test**

Append:

```python
def test_optimizer_activates_v2_loss():
    source = SCRIPT.read_text(encoding="utf-8")
    active = [
        line.strip() for line in source.splitlines()
        if line.strip().startswith("Loss=") and not line.lstrip().startswith("#")
    ]
    assert any("intensityOptLoss_v2(" in line for line in active)
```

- [ ] **Step 2: Run the activation test and verify failure**

Run:

```powershell
pytest tests/test_corner_loss.py::test_optimizer_activates_v2_loss -v
```

Expected: FAIL because the active assignment still calls `intensityOptLoss`.

- [ ] **Step 3: Switch the one-line loss selection**

Replace the active assignment with:

```python
    #Loss=intensityOptLoss(DoC[tstepT0], DoC[tstepT1], DoC[tstepT2], target=(mask/255)).to(device)
    Loss=intensityOptLoss_v2(
        DoC[tstepT0], DoC[tstepT1], DoC[tstepT2], target=(mask/255)
    ).to(device)
```

- [ ] **Step 4: Run focused tests and syntax validation**

Run:

```powershell
pytest tests/test_corner_loss.py -v
python -m py_compile AIE_TEMPOv1.1.py
git diff --check
```

Expected: all tests PASS, compilation succeeds, and `git diff --check` reports no errors.

- [ ] **Step 5: Review for regressions**

Inspect:

```powershell
git diff -- AIE_TEMPOv1.1.py tests/test_corner_loss.py
git status --short
```

Confirm that:

- Existing user changes outside the loss section remain untouched.
- `intensityOptLoss` is byte-for-byte unchanged.
- Only the intended v2 call is active.
- No simulation outputs or unrelated files are staged.

- [ ] **Step 6: Commit activation**

```powershell
git add AIE_TEMPOv1.1.py tests/test_corner_loss.py
git commit -m "feat: activate sharp-corner optimization loss"
```
