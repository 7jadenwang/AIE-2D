"""Differentiable state-space form of the legacy AIE curing model.

The inhibition, diffusion, optics, Dose, and static-mask DoC behavior mirror
``AIE_TEMPOv1.1.py`` and every effective calibration is loaded through the
read-only AST adapter in ``aie_reference.py``. For time-varying MPC masks, a
cumulative reaction-progress state extends the static DoC law without
reinterpreting historical Dose. Projector masks are normalized to ``[0, 1]``;
the adapter validates the legacy 255-level grayscale convention without
importing or executing the reference script.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from aie_fine_grid import expand_projector_mask
from aie_reference import load_reference_config


DOC_HISTORY_MODE = "incremental_reaction_progress_v1"
DOC_HISTORY_DESCRIPTION = (
    "MPC-side time-varying extension; exactly reduces to the reference B/intensity "
    "DoC law at constant intensity and is not separately experimentally calibrated"
)


@dataclass(frozen=True)
class AIEParameters:
    """Physical/model parameters resolved from ``AIE_TEMPOv1.1.py``.

    All inhibitor quantities and accumulated dose are in mJ/cm^2.  The
    projector intensity is in mW/cm^2, so multiplying it by ``dt`` gives the
    per-step energy used by the legacy model. There are intentionally no
    physical defaults here: normal construction must use ``from_reference``.
    """

    pixel_pitch_m: float
    native_pixel_pitch_m: float
    native_shape: tuple[int, int]
    projector_refinement: int
    dt: float
    total_simulation_time_s: float
    loss_sample_times_s: tuple[float, float, float]
    intensity_mw_cm2: float
    tempo_concentration_mM: float | None
    o2_diffusivity_m2_s: float
    tempo_diffusivity_m2_s: float
    o2_inhibition_mj_cm2: float
    total_inhibition_mj_cm2: float
    scattering_blur_size_m: float
    tempo_gaussian_sigma_scale: float
    o2_diffusion_enabled: bool
    tempo_diffusion_enabled: bool
    chain_growth_noise_std: float
    chain_growth_noise_enabled: bool
    b_slope: float
    b_intercept: float
    b_condition_label: str
    minimum_normalized_intensity: float
    division_epsilon: float
    mask_grayscale_max: float
    reference_model_source: str
    reference_model_path: str
    reference_model_sha256: str
    reference_structure_sha256: str
    model_structure_version: int
    doc_model_id: str
    doc_model_formula: str
    doc_fit_applied_to_governing_law: bool
    doc_calibration_source: str
    doc_calibration_path: str
    doc_calibration_sha256: str
    doc_fit_selection_status: str
    available_doc_fit_condition_ids: tuple[str, ...]
    doc_fit_calibration_id: str | None
    doc_fit_condition_id: str | None
    doc_fit_condition_label: str | None
    doc_fit_formula_id: str | None
    doc_fit_formula: str | None
    doc_fit_exported_at_utc: str | None
    doc_fit_a: float | None
    doc_fit_b: float | None
    doc_fit_c: float | None

    @classmethod
    def from_reference(cls, reference: object | None = None) -> "AIEParameters":
        """Map the authoritative reference configuration into model fields."""

        resolved = reference or load_reference_config()
        doc_fit = resolved.doc_fit
        return cls(
            pixel_pitch_m=resolved.native_pixel_pitch_m,
            native_pixel_pitch_m=resolved.native_pixel_pitch_m,
            native_shape=tuple(resolved.native_shape),
            projector_refinement=resolved.projector_refinement,
            dt=resolved.dt,
            total_simulation_time_s=resolved.total_simulation_time_s,
            loss_sample_times_s=tuple(resolved.loss_sample_times_s),
            intensity_mw_cm2=resolved.intensity_mw_cm2,
            tempo_concentration_mM=resolved.tempo_concentration_mM,
            o2_diffusivity_m2_s=resolved.o2_diffusivity_m2_s,
            tempo_diffusivity_m2_s=resolved.tempo_diffusivity_m2_s,
            o2_inhibition_mj_cm2=resolved.o2_inhibition_mj_cm2,
            total_inhibition_mj_cm2=resolved.total_inhibition_mj_cm2,
            scattering_blur_size_m=resolved.scattering_blur_size_m,
            tempo_gaussian_sigma_scale=resolved.tempo_gaussian_sigma_scale,
            o2_diffusion_enabled=resolved.o2_diffusion_enabled,
            tempo_diffusion_enabled=resolved.tempo_diffusion_enabled,
            chain_growth_noise_std=resolved.chain_growth_noise_std,
            chain_growth_noise_enabled=resolved.chain_growth_noise_enabled,
            b_slope=resolved.b_slope,
            b_intercept=resolved.b_intercept,
            b_condition_label=resolved.b_condition_label,
            minimum_normalized_intensity=resolved.minimum_normalized_intensity,
            division_epsilon=resolved.division_epsilon,
            mask_grayscale_max=resolved.mask_grayscale_max,
            reference_model_source=resolved.reference_model_source,
            reference_model_path=resolved.reference_model_path,
            reference_model_sha256=resolved.reference_model_sha256,
            reference_structure_sha256=resolved.reference_structure_sha256,
            model_structure_version=resolved.model_structure_version,
            doc_model_id=resolved.doc_model_id,
            doc_model_formula=resolved.doc_model_formula,
            doc_fit_applied_to_governing_law=(
                resolved.doc_fit_applied_to_governing_law
            ),
            doc_calibration_source=resolved.doc_calibration_source,
            doc_calibration_path=resolved.doc_calibration_path,
            doc_calibration_sha256=resolved.doc_calibration_sha256,
            doc_fit_selection_status=resolved.doc_fit_selection_status,
            available_doc_fit_condition_ids=tuple(
                resolved.available_doc_fit_condition_ids
            ),
            doc_fit_calibration_id=(doc_fit.calibration_id if doc_fit else None),
            doc_fit_condition_id=(doc_fit.condition_id if doc_fit else None),
            doc_fit_condition_label=(doc_fit.condition_label if doc_fit else None),
            doc_fit_formula_id=(doc_fit.formula_id if doc_fit else None),
            doc_fit_formula=(doc_fit.formula if doc_fit else None),
            doc_fit_exported_at_utc=(doc_fit.exported_at_utc if doc_fit else None),
            doc_fit_a=(doc_fit.a if doc_fit else None),
            doc_fit_b=(doc_fit.b if doc_fit else None),
            doc_fit_c=(doc_fit.c if doc_fit else None),
        )

    def provenance_metadata(self) -> dict[str, object]:
        """Return the effective reference values needed to audit a run."""

        return {
            "reference_model_source": self.reference_model_source,
            "reference_model_path": self.reference_model_path,
            "reference_model_sha256": self.reference_model_sha256,
            "reference_structure_sha256": self.reference_structure_sha256,
            "model_structure_version": self.model_structure_version,
            "native_shape": self.native_shape,
            "native_pixel_pitch_m": self.native_pixel_pitch_m,
            "effective_pixel_pitch_m": self.pixel_pitch_m,
            "projector_refinement": self.projector_refinement,
            "dt": self.dt,
            "total_simulation_time_s": self.total_simulation_time_s,
            "loss_sample_times_s": self.loss_sample_times_s,
            "intensity_mw_cm2": self.intensity_mw_cm2,
            "tempo_concentration_mM": self.tempo_concentration_mM,
            "o2_diffusivity_m2_s": self.o2_diffusivity_m2_s,
            "tempo_diffusivity_m2_s": self.tempo_diffusivity_m2_s,
            "o2_inhibition_mj_cm2": self.o2_inhibition_mj_cm2,
            "tempo_inhibition_mj_cm2": self.tempo_inhibition_mj_cm2,
            "total_inhibition_mj_cm2": self.total_inhibition_mj_cm2,
            "scattering_blur_size_m": self.scattering_blur_size_m,
            "tempo_gaussian_sigma_scale": self.tempo_gaussian_sigma_scale,
            "o2_diffusion_enabled": self.o2_diffusion_enabled,
            "tempo_diffusion_enabled": self.tempo_diffusion_enabled,
            "chain_growth_noise_std": self.chain_growth_noise_std,
            "chain_growth_noise_enabled": self.chain_growth_noise_enabled,
            "b_slope": self.b_slope,
            "b_intercept": self.b_intercept,
            "b_condition_label": self.b_condition_label,
            "minimum_normalized_intensity": self.minimum_normalized_intensity,
            "division_epsilon": self.division_epsilon,
            "mask_grayscale_max": self.mask_grayscale_max,
            "doc_model_id": self.doc_model_id,
            "doc_model_formula": self.doc_model_formula,
            "doc_history_mode": DOC_HISTORY_MODE,
            "doc_history_description": DOC_HISTORY_DESCRIPTION,
            "doc_fit_applied_to_governing_law": (
                self.doc_fit_applied_to_governing_law
            ),
            "doc_calibration_source": self.doc_calibration_source,
            "doc_calibration_path": self.doc_calibration_path,
            "doc_calibration_sha256": self.doc_calibration_sha256,
            "doc_fit_selection_status": self.doc_fit_selection_status,
            "available_doc_fit_condition_ids": self.available_doc_fit_condition_ids,
            "doc_fit_calibration_id": self.doc_fit_calibration_id,
            "doc_fit_condition_id": self.doc_fit_condition_id,
            "doc_fit_condition_label": self.doc_fit_condition_label,
            "doc_fit_formula_id": self.doc_fit_formula_id,
            "doc_fit_formula": self.doc_fit_formula,
            "doc_fit_exported_at_utc": self.doc_fit_exported_at_utc,
            "doc_fit_a": self.doc_fit_a,
            "doc_fit_b": self.doc_fit_b,
            "doc_fit_c": self.doc_fit_c,
        }

    def __post_init__(self) -> None:
        positive = {
            "pixel_pitch_m": self.pixel_pitch_m,
            "native_pixel_pitch_m": self.native_pixel_pitch_m,
            "dt": self.dt,
            "total_simulation_time_s": self.total_simulation_time_s,
            "intensity_mw_cm2": self.intensity_mw_cm2,
            "o2_diffusivity_m2_s": self.o2_diffusivity_m2_s,
            "tempo_diffusivity_m2_s": self.tempo_diffusivity_m2_s,
            "scattering_blur_size_m": self.scattering_blur_size_m,
            "tempo_gaussian_sigma_scale": self.tempo_gaussian_sigma_scale,
            "minimum_normalized_intensity": self.minimum_normalized_intensity,
            "division_epsilon": self.division_epsilon,
            "mask_grayscale_max": self.mask_grayscale_max,
        }
        for name, value in positive.items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        finite_values = {
            "o2_inhibition_mj_cm2": self.o2_inhibition_mj_cm2,
            "total_inhibition_mj_cm2": self.total_inhibition_mj_cm2,
            "chain_growth_noise_std": self.chain_growth_noise_std,
            "b_slope": self.b_slope,
            "b_intercept": self.b_intercept,
        }
        if self.tempo_concentration_mM is not None:
            finite_values["tempo_concentration_mM"] = self.tempo_concentration_mM
        doc_fit_values = (self.doc_fit_a, self.doc_fit_b, self.doc_fit_c)
        if any(value is not None for value in doc_fit_values) and not all(
            value is not None for value in doc_fit_values
        ):
            raise ValueError("DoC fit a/b/c must either all be present or all be absent")
        if self.doc_fit_a is not None:
            finite_values.update(
                {
                    "doc_fit_a": self.doc_fit_a,
                    "doc_fit_b": self.doc_fit_b,
                    "doc_fit_c": self.doc_fit_c,
                }
            )
        for name, value in finite_values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if (
            (self.tempo_concentration_mM is not None and self.tempo_concentration_mM < 0)
            or self.o2_inhibition_mj_cm2 < 0
            or self.total_inhibition_mj_cm2 < 0
            or self.chain_growth_noise_std < 0
        ):
            raise ValueError(
                "TEMPO concentration, noise, and inhibition energies must be nonnegative"
            )
        if self.projector_refinement < 1:
            raise ValueError("projector_refinement must be at least 1")
        if len(self.native_shape) != 2 or any(value < 1 for value in self.native_shape):
            raise ValueError(f"native_shape must be two positive dimensions, got {self.native_shape}")
        if self.model_structure_version < 1:
            raise ValueError("model_structure_version must be at least 1")
        for name, fingerprint in (
            ("reference_model_sha256", self.reference_model_sha256),
            ("reference_structure_sha256", self.reference_structure_sha256),
            ("doc_calibration_sha256", self.doc_calibration_sha256),
        ):
            if len(fingerprint) != 64 or any(
                character not in "0123456789abcdef" for character in fingerprint
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")

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
    reaction_progress: torch.Tensor
    doc: torch.Tensor
    chain_growth_multiplier: torch.Tensor | None = None

    @property
    def shape(self) -> torch.Size:
        return self.o2.shape

    def detach(self) -> "AIEState":
        """Return a state detached at an MPC estimation boundary."""

        multiplier = (
            self.chain_growth_multiplier.detach()
            if self.chain_growth_multiplier is not None
            else None
        )
        return AIEState(
            o2=self.o2.detach(),
            tempo=self.tempo.detach(),
            dose=self.dose.detach(),
            reaction_progress=self.reaction_progress.detach(),
            doc=self.doc.detach(),
            chain_growth_multiplier=multiplier,
        )

    def tensors(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return every spatial dynamic-state field, including cure history."""

        return self.o2, self.tempo, self.dose, self.reaction_progress, self.doc

    def reference_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fields that also exist in the collaborator reference model."""

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
        self.params = params if params is not None else AIEParameters.from_reference()
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
            tempo_size,
            tempo_sigma * self.params.tempo_gaussian_sigma_scale,
            device=selected_device,
            dtype=dtype,
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
        chain_growth_multiplier = None
        if self.params.chain_growth_noise_std > 0:
            chain_growth_multiplier = (
                1
                + self.params.chain_growth_noise_std
                * torch.randn(spatial_shape, **field_options)
            ).clamp_min(1e-3)
        return AIEState(
            o2=o2,
            tempo=tempo,
            dose=zeros.clone(),
            reaction_progress=zeros.clone(),
            doc=zeros.clone(),
            chain_growth_multiplier=chain_growth_multiplier,
        )

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
        """Advance the physical state by one reference-configured time step."""

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

        o2_diffused = (
            self._gaussian_blur(state.o2, self.o2_kernel_1d, "O2 diffusion")
            if self.params.o2_diffusion_enabled
            else state.o2
        )
        tempo_diffused = (
            self._gaussian_blur(
                state.tempo, self.tempo_kernel_1d, "TEMPO diffusion"
            )
            if self.params.tempo_diffusion_enabled
            else state.tempo
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
        # The reference Dose bookkeeping is retained verbatim above.  Cure
        # history uses the increment produced by this step rather than
        # reinterpreting all historical Dose through the current intensity.
        delta_dose = dose_next - state.dose
        safe_intensity = control.local_intensity.clamp_min(
            self.params.division_epsilon
        )
        effective_b = control.b
        if state.chain_growth_multiplier is not None:
            effective_b = effective_b * state.chain_growth_multiplier
        reaction_increment = effective_b * delta_dose / safe_intensity
        reaction_progress_candidate = (
            state.reaction_progress + reaction_increment
        )
        reaction_progress_next = torch.where(
            curing,
            reaction_progress_candidate,
            state.reaction_progress,
        )
        doc_candidate = 1.0 - torch.exp(
            -torch.clamp(reaction_progress_next, min=0.0)
        )
        doc_next = torch.where(curing, doc_candidate, state.doc)
        return AIEState(
            o2=o2_next,
            tempo=tempo_next,
            dose=dose_next,
            reaction_progress=reaction_progress_next,
            doc=doc_next,
            chain_growth_multiplier=state.chain_growth_multiplier,
        )

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
        for name, field in zip(
            ("o2", "tempo", "dose", "reaction_progress", "doc"),
            state.tensors(),
        ):
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
        multiplier = state.chain_growth_multiplier
        if self.params.chain_growth_noise_std > 0 and multiplier is None:
            raise ValueError(
                "state.chain_growth_multiplier is required when "
                "chain_growth_noise_std > 0"
            )
        if multiplier is not None:
            if tuple(multiplier.shape) != reference_shape:
                raise ValueError(
                    "state.chain_growth_multiplier has shape "
                    f"{tuple(multiplier.shape)}, expected {reference_shape}"
                )
            if multiplier.device != self.device or multiplier.dtype != self.dtype:
                raise ValueError(
                    "state.chain_growth_multiplier must match model device and dtype"
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
