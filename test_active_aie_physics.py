"""Small CPU physics checks; no MPC optimization or authoritative script import.

Prepared for a later authorized run: python -m unittest test_active_aie_physics.
The oracle evaluates selected ACTIVE source expressions, not a copied B fit.
Time-varying R is an explicit controller extension, not collaborator validation.
"""
import ast
from dataclasses import replace
from pathlib import Path
import unittest

import torch

from aie_model import AIEModel, AIEParameters
from aie_reference import load_reference_config, reference_step_torch


class ActivePhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = load_reference_config()
        cls.model = AIEModel(
            params=AIEParameters.from_reference(cls.reference),
            device=torch.device("cpu"), dtype=torch.float64,
        )
        size = max(k.numel() for k in (
            cls.model.scattering_kernel_1d, cls.model.o2_kernel_1d,
            cls.model.tempo_kernel_1d,
        )) // 2 + 3
        cls.shape = (size, size + 2)
        tree = ast.parse(Path(cls.reference.reference_model_path).read_text(encoding="utf-8-sig"))
        cls.expressions = {}
        for node in sorted(ast.walk(tree), key=lambda item: getattr(item, "lineno", -1)):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        cls.expressions.setdefault(target.id, []).append(node.value)

    def expression(self, name, env, index=-1):
        return eval(compile(ast.Expression(self.expressions[name][index]),
                            self.reference.reference_model_path, "eval"),
                    {"torch": torch}, env)

    def oracle(self, state, mask):
        # Full 2D convolutions independently mirror collaborator diffusion/optics.
        import torch.nn.functional as F
        def blur(field, kernel):
            pad = kernel.shape[-1] // 2
            if not pad:
                return field
            return F.conv2d(F.pad(field[None, None], (pad,) * 4, mode="reflect"),
                            kernel)[0, 0]
        r = self.reference
        env = dict(
            blur_mask=blur(mask, self.model.scattering_kernel_2d) * r.mask_grayscale_max,
            intensity=r.intensity_mw_cm2, dt=r.dt,
            pre_slope=r.pre_slope, pre_cap=r.pre_cap,
            O2_diffused=blur(state.o2, self.model.o2_kernel_2d) if r.o2_diffusion_enabled else state.o2,
            TEMPO_diffused=blur(state.tempo, self.model.tempo_kernel_2d) if r.tempo_diffusion_enabled else state.tempo,
            Dose=[state.dose], cum_light=state.cumulative_light,
        )
        for name in ("local_I", "B", "energy", "O2next", "energy_after_o2",
                     "TEMPOnext", "energy_after_inhibition", "Dosenext", "t",
                     "cum_light", "pre_cure", "DoCnext"):
            env[name] = self.expression(name, env)
        return env

    def assert_fields(self, actual, oracle):
        for field, name in ((actual.o2, "O2next"), (actual.tempo, "TEMPOnext"),
                            (actual.dose, "Dosenext"), (actual.doc, "DoCnext"),
                            (actual.cumulative_light, "cum_light")):
            torch.testing.assert_close(field, oracle[name], rtol=1e-9, atol=1e-11)

    def test_rate_residual_energy_static_and_reference_fixture(self):
        mask = torch.linspace(0.1, 0.9, self.shape[0] * self.shape[1],
                              dtype=torch.float64).reshape(self.shape)
        state = self.model.initialize_state(self.shape)
        energy = self.model.prepare_control(mask, self.shape).energy
        # Exercise both partial depletion and crossing into cure in a few steps.
        state = replace(state, o2=energy * 1.2, tempo=energy * 0.7)
        for _ in range(5):
            expected = self.oracle(state, mask)
            prepared = self.model.prepare_control(mask, self.shape)
            torch.testing.assert_close(prepared.b, expected["B"], rtol=1e-10, atol=1e-12)
            independent = reference_step_torch(
                o2=state.o2, tempo=state.tempo, dose=state.dose, doc=state.doc,
                cumulative_light=state.cumulative_light, normalized_mask=mask,
                scattering_kernel_2d=self.model.scattering_kernel_2d,
                o2_kernel_2d=self.model.o2_kernel_2d,
                tempo_kernel_2d=self.model.tempo_kernel_2d, config=self.reference,
            )
            actual = self.model.step(state, mask)
            self.assert_fields(actual, expected)
            for name in ("o2", "tempo", "dose", "doc", "cumulative_light"):
                torch.testing.assert_close(getattr(actual, name), getattr(independent, name),
                                           rtol=1e-9, atol=1e-11)
            self.assertTrue(bool((actual.dose >= state.dose).all()))
            state = actual

    def test_active_zero_tempo_has_no_extra_curing_delay(self):
        self.assertEqual(self.reference.tempo_concentration_mM, 0.0)
        self.assertEqual(self.reference.total_inhibition_mj_cm2,
                         self.reference.o2_inhibition_mj_cm2)
        self.assertEqual(self.reference.tempo_inhibition_mj_cm2, 0.0)
        self.assertEqual(self.reference.b_law_type, "power_law")
        self.assertTrue(self.reference.o2_diffusion_enabled)
        mask = torch.ones(self.shape, dtype=torch.float64)
        state = self.model.initialize_state(self.shape)
        torch.testing.assert_close(state.tempo, torch.zeros_like(mask), rtol=0, atol=0)
        torch.testing.assert_close(state.o2, torch.full_like(mask, self.reference.o2_inhibition_mj_cm2))
        prepared = self.model.prepare_control(mask, self.shape)
        # Cross O2 depletion in one step; TEMPO must neither consume nor delay.
        state = replace(state, o2=prepared.energy * 0.5)
        actual = self.model.step_prepared(state, prepared)
        self.assert_fields(actual, self.oracle(state, mask))
        torch.testing.assert_close(actual.tempo, torch.zeros_like(mask), rtol=0, atol=0)
        torch.testing.assert_close(actual.dose, prepared.energy * 0.5)
        self.assertTrue(bool((actual.reaction_progress > 0).all()))
        torch.testing.assert_close(actual.doc, 1 - torch.exp(-actual.reaction_progress))

    def test_precure_is_not_reaction_progress_and_survives_detach(self):
        mask = torch.full(self.shape, 0.5, dtype=torch.float64)
        state = self.model.initialize_state(self.shape)
        state = replace(state, o2=torch.full_like(mask, 1e6),
                        cumulative_light=torch.full_like(mask, self.reference.pre_cap / self.reference.pre_slope))
        expected = self.oracle(state, mask)
        actual = self.model.step(state.detach(), mask)
        self.assert_fields(actual, expected)
        torch.testing.assert_close(actual.reaction_progress, torch.zeros_like(mask))
        torch.testing.assert_close(actual.doc, torch.full_like(mask, self.reference.pre_cap))

    def test_time_varying_control_retains_incremental_history(self):
        state = self.model.initialize_state(self.shape)
        state = replace(state, o2=torch.zeros_like(state.o2), tempo=torch.zeros_like(state.tempo))
        expected_r = torch.zeros_like(state.doc)
        for level in (0.8, 0.2, 0.6):
            mask = torch.full(self.shape, level, dtype=torch.float64)
            source = self.oracle(state, mask)
            actual = self.model.step(state, mask)
            expected_r = expected_r + source["B"] * (source["Dosenext"] - state.dose) / source["local_I"]
            torch.testing.assert_close(actual.reaction_progress, expected_r)
            torch.testing.assert_close(actual.doc, -torch.expm1(-expected_r))
            for name, key in (("o2", "O2next"), ("tempo", "TEMPOnext"), ("dose", "Dosenext")):
                torch.testing.assert_close(getattr(actual, name), source[key])
            state = actual
        # Historical dose must not be reweighted using the final mask's B/I.
        self.assertFalse(torch.allclose(actual.doc, source["DoCnext"], rtol=1e-6, atol=1e-10))

    def test_low_intensity_gradients_and_authoritative_dark_floor(self):
        for level in (0.0, 1e-14, 1e-8):
            mask = torch.full(self.shape, level, dtype=torch.float64, requires_grad=True)
            state = self.model.initialize_state(self.shape)
            state = replace(state, o2=torch.zeros_like(state.o2), tempo=torch.zeros_like(state.tempo))
            actual = self.model.step(state, mask)
            gradient, = torch.autograd.grad(actual.doc.sum(), mask)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertTrue(bool(torch.isfinite(actual.doc).all()))
            if level == 0:
                # Source clamps grayscale before intensity conversion. No extra
                # cure floor is introduced: only this source-level dark residual.
                expected = self.oracle(state, mask)
                self.assert_fields(actual, expected)
                self.assertLess(float(actual.doc.max()), 1e-8)


if __name__ == "__main__":
    unittest.main()
