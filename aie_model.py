"""Differentiable state-space form of the legacy AIE curing model.

The equations and default calibration in this module are taken from
``AIE_TEMPOv1.1.py``.  Projector masks are represented in normalized
grayscale units in ``[0, 1]``; multiplying by 255 recovers the convention
used by that script before its ``/ 255`` operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from aie_fine_grid import expand_projector_mask


@dataclass(frozen=True)
class AIEParameters:
    """Physical and numerical parameters from ``AIE_TEMPOv1.1.py``.

    All inhibitor quantities and accumulated dose are in mJ/cm^2.  The
    projector intensity is in mW/cm^2, so multiplying it by ``dt`` gives the
    per-step energy used by the legacy model.
    """

    pixel_pitch_m: float = 4.905e-6
    dt: float = 0.025
    intensity_mw_cm2: float = 20.0
    o2_diffusivity_m2_s: float = 2000e-12
    tempo_diffusivity_m2_s: float = 400e-12
    o2_inhibition_mj_cm2: float = 27.5100
    total_inhibition_mj_cm2: float = 0.0
    scattering_blur_size_m: float = 600e-6
    b_slope: float = 0.0163
    b_intercept: float = 0.4148
    projector_refinement: int = 1
    minimum_normalized_intensity: float = 1e-12
    division_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        positive = {
            "pixel_pitch_m": self.pixel_pitch_m,
            "dt": self.dt,
            "intensity_mw_cm2": self.intensity_mw_cm2,
            "o2_diffusivity_m2_s": self.o2_diffusivity_m2_s,
            "tempo_diffusivity_m2_s": self.tempo_diffusivity_m2_s,
            "scattering_blur_size_m": self.scattering_blur_size_m,
            "minimum_normalized_intensity": self.minimum_normalized_intensity,
            "division_epsilon": self.division_epsilon,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.o2_inhibition_mj_cm2 < 0 or self.total_inhibition_mj_cm2 < 0:
            raise ValueError("inhibition energies must be nonnegative")
        if self.projector_refinement < 1:
            raise ValueError("projector_refinement must be at least 1")

    @property
    def tempo_inhibition_mj_cm2(self) -> float:
        """TEMPO inhibition as calculated in the legacy script."""

        return max(0.0, self.total_inhibition_mj_cm2 - self.o2_inhibition_mj_cm2)


@dataclass
class AIEState:
    """Spatial state fields advanced by one physical time step at a time."""

    o2: torch.Tensor
    tempo: torch.Tensor
    dose: torch.Tensor
    doc: torch.Tensor

    @property
    def shape(self) -> torch.Size:
        return self.o2.shape

    def detach(self) -> "AIEState":
        """Return a state detached at an MPC estimation boundary."""

        return AIEState(*(field.detach() for field in self.tensors()))

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.o2, self.tempo, self.dose, self.doc


@dataclass(frozen=True)
class AIEPreparedControl:
    """Optical quantities that stay fixed while one DLP mask is held."""

    normalized_mask: torch.Tensor
    scattered_mask: torch.Tensor
    local_intensity: torch.Tensor
    energy: torch.Tensor
    b: torch.Tensor


def _legacy_diffusion_kernel_size(sigma_pixels: float) -> int:
    """Reproduce the empirical kernel-size expression in the legacy model."""

    return max(1, int((sigma_pixels - 0.8) / 0.3 + 1) * 2 + 1)


def _gaussian_kernel_1d(
    kernel_size: int,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Match ``cv2.getGaussianKernel(size, sigma)`` without requiring OpenCV."""

    # OpenCV creates its explicitly parameterized kernel in double precision;
    # doing the same before casting minimizes differences from v1.1.
    coordinates = torch.arange(kernel_size, dtype=torch.float64)
    coordinates = coordinates - (kernel_size - 1) / 2
    kernel = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.to(device=device, dtype=dtype)


class AIEModel(nn.Module):
    """Differentiable nonlinear AIE vat-photopolymerization predictor.

    ``step`` is the state transition ``x[k+1] = f(x[k], u[k])``.  Use
    ``advance`` when a mask is held for multiple physics steps; it computes
    the unchanged optical scattering only once while preserving the exact
    physical update and its autograd graph.
    """

    def __init__(
        self,
        params: AIEParameters | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.params = params or AIEParameters()
        selected_device = torch.device(device or "cpu")
        if not dtype.is_floating_point:
            raise TypeError(f"AIEModel requires a floating dtype, got {dtype}")

        o2_sigma = sqrt(
            2 * self.params.o2_diffusivity_m2_s * self.params.dt
        ) / self.params.pixel_pitch_m
        tempo_sigma = sqrt(
            2 * self.params.tempo_diffusivity_m2_s * self.params.dt
        ) / self.params.pixel_pitch_m
        o2_size = _legacy_diffusion_kernel_size(o2_sigma)
        tempo_size = _legacy_diffusion_kernel_size(tempo_sigma)

        blur_pixels = int(
            self.params.scattering_blur_size_m / self.params.pixel_pitch_m
        )
        scattering_size = blur_pixels if blur_pixels % 2 else blur_pixels + 1
        scattering_sigma = 0.3 * ((scattering_size - 1) * 0.5 - 1) + 0.8

        o2_kernel = _gaussian_kernel_1d(
            o2_size, o2_sigma, device=selected_device, dtype=dtype
        )
        tempo_kernel = _gaussian_kernel_1d(
            tempo_size, tempo_sigma, device=selected_device, dtype=dtype
        )
        scattering_kernel = _gaussian_kernel_1d(
            scattering_size,
            scattering_sigma,
            device=selected_device,
            dtype=dtype,
        )
        self.register_buffer("o2_kernel_1d", o2_kernel)
        self.register_buffer("tempo_kernel_1d", tempo_kernel)
        self.register_buffer("scattering_kernel_1d", scattering_kernel)
        # The 2-D buffers are useful for direct comparisons with the legacy
        # cv2/conv2d implementation in validation code.
        self.register_buffer(
            "o2_kernel_2d", torch.outer(o2_kernel, o2_kernel)[None, None]
        )
        self.register_buffer(
            "tempo_kernel_2d", torch.outer(tempo_kernel, tempo_kernel)[None, None]
        )
        self.register_buffer(
            "scattering_kernel_2d",
            torch.outer(scattering_kernel, scattering_kernel)[None, None],
        )

    @property
    def device(self) -> torch.device:
        return self.scattering_kernel_1d.device

    @property
    def dtype(self) -> torch.dtype:
        return self.scattering_kernel_1d.dtype

    @property
    def scattering_kernel_size(self) -> int:
        return int(self.scattering_kernel_1d.numel())

    def initialize_state(self, shape: Sequence[int]) -> AIEState:
        """Create the legacy uniform-inhibitor, zero-dose initial condition."""

        spatial_shape = self._normalize_shape(shape)
        self._validate_spatial_shape(spatial_shape)
        field_options = {"device": self.device, "dtype": self.dtype}
        o2 = torch.full(
            spatial_shape, self.params.o2_inhibition_mj_cm2, **field_options
        )
        tempo = torch.full(
            spatial_shape, self.params.tempo_inhibition_mj_cm2, **field_options
        )
        zeros = torch.zeros(spatial_shape, **field_options)
        return AIEState(o2=o2, tempo=tempo, dose=zeros.clone(), doc=zeros.clone())

    def control_shape_for(self, state_shape: Sequence[int]) -> tuple[int, int]:
        """Return the DLP mask shape corresponding to a simulation-grid shape."""

        height, width = self._normalize_shape(state_shape)
        refinement = self.params.projector_refinement
        if height % refinement or width % refinement:
            raise ValueError(
                f"state shape {(height, width)} must be divisible by projector_refinement "
                f"{refinement}"
            )
        return height // refinement, width // refinement

    def prepare_control(
        self,
        projector_mask: torch.Tensor,
        state_shape: Sequence[int],
    ) -> AIEPreparedControl:
        """Expand and scatter one normalized projector mask.

        This method is intentionally differentiable.  It can be called once
        per control interval and its output reused by ``step_prepared``.
        """

        spatial_shape = self._normalize_shape(state_shape)
        expected_control_shape = self.control_shape_for(spatial_shape)
        self._validate_mask(projector_mask, expected_control_shape)
        fine_mask = (
            expand_projector_mask(projector_mask, self.params.projector_refinement)
            if self.params.projector_refinement > 1
            else projector_mask
        )
        scattered_mask = self._gaussian_blur(
            fine_mask, self.scattering_kernel_1d, "scattering"
        )
        normalized_intensity = scattered_mask.clamp_min(
            self.params.minimum_normalized_intensity
        )
        local_intensity = normalized_intensity * self.params.intensity_mw_cm2
        energy = local_intensity * self.params.dt
        b = self.params.b_slope * local_intensity + self.params.b_intercept
        return AIEPreparedControl(
            normalized_mask=projector_mask,
            scattered_mask=scattered_mask,
            local_intensity=local_intensity,
            energy=energy,
            b=b,
        )

    def step(self, state: AIEState, projector_mask: torch.Tensor) -> AIEState:
        """Advance the physical state by one 0.025 s step by default."""

        self._validate_state(state)
        prepared = self.prepare_control(projector_mask, state.shape)
        return self.step_prepared(state, prepared)

    def step_prepared(
        self, state: AIEState, control: AIEPreparedControl
    ) -> AIEState:
        """Advance one step using already-computed optical quantities."""

        self._validate_state(state)
        if tuple(control.energy.shape) != tuple(state.shape):
            raise ValueError(
                f"prepared control has shape {tuple(control.energy.shape)}, expected "
                f"state shape {tuple(state.shape)}"
            )

        # v1.1 computes an O2 convolution and immediately overwrites it with
        # the previous local field.  Preserve that effective no-diffusion
        # behavior instead of silently enabling O2 diffusion.
        o2_diffused = state.o2

        # The legacy source marks this Gaussian TEMPO diffusion expression as
        # questionable ("wrong here") but uses it, so it remains unchanged.
        tempo_diffused = self._gaussian_blur(
            state.tempo, self.tempo_kernel_1d, "TEMPO diffusion"
        )

        o2_next = torch.clamp(o2_diffused - control.energy, min=0.0)
        tempo_next = torch.where(
            o2_next <= 0,
            torch.clamp(tempo_diffused - control.energy, min=0.0),
            tempo_diffused,
        )
        curing = (o2_next <= 0) & (tempo_next <= 0)

        # Preserve the v1.1 first-curing-step bookkeeping exactly.  In a cell
        # where both inhibitors cross zero together this subtracts both
        # pre-step fields and can transiently make dose negative.
        dose_next = torch.where(
            curing,
            state.dose + control.energy - o2_diffused - tempo_diffused,
            state.dose,
        )
        # The legacy denominator is intensity clamped through the mask at
        # 1e-12.  The second clamp guarantees finite behavior for any valid
        # floating dtype and avoids the legacy zero-intensity NaN/Inf hazard.
        safe_intensity = control.local_intensity.clamp_min(
            self.params.division_epsilon
        )
        # With a time-varying mask, v1.1's static-mask formula necessarily
        # interprets accumulated dose using the *current* local intensity.
        # Keep that literal extension here rather than inventing a new kinetic
        # state or exposure law for this first controller.
        exposure_time = dose_next / safe_intensity
        doc_candidate = 1.0 - torch.exp(
            -torch.clamp(control.b * exposure_time, min=0.0)
        )
        doc_next = torch.where(curing, doc_candidate, state.doc)
        return AIEState(o2=o2_next, tempo=tempo_next, dose=dose_next, doc=doc_next)

    def advance(
        self,
        state: AIEState,
        projector_mask: torch.Tensor,
        physics_steps: int = 1,
    ) -> AIEState:
        """Hold one mask fixed and advance several physical time steps."""

        if physics_steps < 1:
            raise ValueError(f"physics_steps must be at least 1, got {physics_steps}")
        self._validate_state(state)
        prepared = self.prepare_control(projector_mask, state.shape)
        next_state = state
        for _ in range(physics_steps):
            next_state = self.step_prepared(next_state, prepared)
        return next_state

    def rollout(
        self,
        initial_state: AIEState,
        control_sequence: torch.Tensor,
        physics_steps_per_control: int = 1,
    ) -> AIEState:
        """Roll out a ``(horizon, mask_height, mask_width)`` control sequence."""

        self._validate_state(initial_state)
        if control_sequence.ndim != 3:
            raise ValueError(
                "control_sequence must have shape (horizon, height, width), got "
                f"{tuple(control_sequence.shape)}"
            )
        if control_sequence.shape[0] < 1:
            raise ValueError("control_sequence horizon must be at least 1")
        expected = self.control_shape_for(initial_state.shape)
        if tuple(control_sequence.shape[1:]) != expected:
            raise ValueError(
                f"control_sequence masks must have shape {expected}, got "
                f"{tuple(control_sequence.shape[1:])}"
            )
        state = initial_state
        for control in control_sequence:
            state = self.advance(state, control, physics_steps_per_control)
        return state

    def _gaussian_blur(
        self, field: torch.Tensor, kernel_1d: torch.Tensor, name: str
    ) -> torch.Tensor:
        """Apply the legacy 2-D Gaussian as two equivalent separable filters."""

        padding = int(kernel_1d.numel() // 2)
        if padding == 0:
            return field
        height, width = field.shape
        if padding >= height or padding >= width:
            raise ValueError(
                f"{name} reflect padding {padding} requires both spatial dimensions "
                f"to exceed {padding}, got {(height, width)}"
            )
        image = field[None, None]
        horizontal = F.conv2d(
            F.pad(image, (padding, padding, 0, 0), mode="reflect"),
            kernel_1d.view(1, 1, 1, -1),
        )
        blurred = F.conv2d(
            F.pad(horizontal, (0, 0, padding, padding), mode="reflect"),
            kernel_1d.view(1, 1, -1, 1),
        )
        return blurred[0, 0]

    def _validate_state(self, state: AIEState) -> None:
        if not isinstance(state, AIEState):
            raise TypeError(f"state must be AIEState, got {type(state).__name__}")
        reference_shape = tuple(state.o2.shape)
        if len(reference_shape) != 2:
            raise ValueError(f"state fields must be 2D, got shape {reference_shape}")
        self._validate_spatial_shape(reference_shape)
        for name, field in zip(("o2", "tempo", "dose", "doc"), state.tensors()):
            if tuple(field.shape) != reference_shape:
                raise ValueError(
                    f"state.{name} has shape {tuple(field.shape)}, expected {reference_shape}"
                )
            if field.device != self.device:
                raise ValueError(
                    f"state.{name} is on {field.device}, but model is on {self.device}"
                )
            if field.dtype != self.dtype:
                raise ValueError(
                    f"state.{name} has dtype {field.dtype}, expected {self.dtype}"
                )

    def _validate_mask(
        self, projector_mask: torch.Tensor, expected_shape: tuple[int, int]
    ) -> None:
        if projector_mask.ndim != 2 or tuple(projector_mask.shape) != expected_shape:
            raise ValueError(
                f"projector_mask must have shape {expected_shape}, got "
                f"{tuple(projector_mask.shape)}"
            )
        if projector_mask.device != self.device:
            raise ValueError(
                f"projector_mask is on {projector_mask.device}, model is on {self.device}"
            )
        if projector_mask.dtype != self.dtype:
            raise ValueError(
                f"projector_mask has dtype {projector_mask.dtype}, expected {self.dtype}"
            )
        # Validation is deliberately outside autograd; the original mask is
        # still used unchanged by every physical operation after this check.
        with torch.no_grad():
            if not bool(torch.isfinite(projector_mask).all()):
                raise ValueError("projector_mask contains NaN or Inf")
            minimum = float(projector_mask.min())
            maximum = float(projector_mask.max())
        tolerance = 1e-6
        if minimum < -tolerance or maximum > 1.0 + tolerance:
            raise ValueError(
                "projector_mask must be normalized to [0, 1], got range "
                f"[{minimum:.6g}, {maximum:.6g}]"
            )

    def _validate_spatial_shape(self, shape: tuple[int, int]) -> None:
        padding = self.scattering_kernel_size // 2
        if shape[0] <= padding or shape[1] <= padding:
            raise ValueError(
                f"state shape {shape} is too small for legacy scattering reflect "
                f"padding {padding}; each dimension must exceed {padding}"
            )

    @staticmethod
    def _normalize_shape(shape: Sequence[int]) -> tuple[int, int]:
        values = tuple(int(value) for value in shape)
        if len(values) != 2 or any(value < 1 for value in values):
            raise ValueError(f"shape must contain two positive dimensions, got {values}")
        return values[0], values[1]
