"""Prepared regression checks; synthetic reconciled sources are memory-only.

No authoritative file edits, result writes, or MPC optimization. Synthetic
conditions test adapter/state invariants, not collaborator calibration validity.
"""
from dataclasses import replace
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import torch

import aie_reference as reference
from aie_model import AIEModel, AIEParameters


def fixture_config(concentration, total_expression, selector="active"):
    original_read = Path.read_bytes
    path = reference.REFERENCE_MODEL_PATH
    source = original_read(path).decode("utf-8")
    source, count = re.subn(r"^Totalinhibtion\s*=.*$",
                           f"Totalinhibtion={total_expression}", source,
                           flags=re.MULTILINE)
    if count != 1:
        raise AssertionError("expected one active total-inhibition assignment")
    # Label and explicit concentration agree in this synthetic source.
    source = re.sub(r"^(\s*B\s*=.*?#\s*)[0-9.]+mMTEMPO",
                    rf"\g<1>{concentration:g}mMTEMPO", source, flags=re.MULTILINE)
    source += f"\nTEMPO_concentration_mM={concentration!r}\n"
    def read_bytes(candidate):
        return source.encode("utf-8") if candidate == path else original_read(candidate)
    with patch.object(Path, "read_bytes", read_bytes):
        return reference.load_reference_config_for_condition(selector)


class InhibitionConditionTests(unittest.TestCase):
    def test_inconsistent_source_and_named_selector_rejected(self):
        for selector in ("active", "30mW_0mM"):
            with self.subTest(selector=selector):
                with self.assertRaisesRegex(reference.ReferenceResolutionError,
                                             "Inconsistent 0 mM.*reconciliation"):
                    fixture_config(0.0, "2 * O2inhibition", selector)

    def test_verified_o2_only_has_no_tempo_delay(self):
        config = fixture_config(0.0, "O2inhibition")
        self.assertEqual(config.total_inhibition_mj_cm2, config.o2_inhibition_mj_cm2)
        self.assertEqual(config.tempo_inhibition_mj_cm2, 0.0)
        model = AIEModel(AIEParameters.from_reference(config), dtype=torch.float64)
        size = max(model.o2_kernel_1d.numel(), model.tempo_kernel_1d.numel(),
                   model.scattering_kernel_1d.numel()) // 2 + 3
        mask = torch.ones((size, size), dtype=torch.float64)
        state = model.initialize_state(mask.shape)
        self.assertEqual(float(state.tempo.max()), 0.0)
        control = model.prepare_control(mask, state.shape)
        # Isolate the O2 crossing without a long rollout.
        state = replace(state, o2=control.energy * 0.5)
        after = model.step_prepared(state, control)
        torch.testing.assert_close(after.tempo, torch.zeros_like(mask), atol=0, rtol=0)
        torch.testing.assert_close(after.dose, control.energy * 0.5)
        self.assertTrue(bool((after.reaction_progress > 0).all()))
        torch.testing.assert_close(after.doc, 1 - torch.exp(-after.reaction_progress))
        with self.assertRaisesRegex(reference.ReferenceResolutionError, "Inconsistent 0 mM"):
            replace(model.params, total_inhibition_mj_cm2=config.o2_inhibition_mj_cm2 + 1)

    def test_nonzero_tempo_capacity_and_delay_preserved(self):
        config = fixture_config(3.0, "2 * O2inhibition")
        self.assertEqual(config.tempo_inhibition_mj_cm2, config.o2_inhibition_mj_cm2)
        model = AIEModel(AIEParameters.from_reference(config), dtype=torch.float64)
        size = max(model.o2_kernel_1d.numel(), model.tempo_kernel_1d.numel(),
                   model.scattering_kernel_1d.numel()) // 2 + 3
        mask = torch.ones((size, size), dtype=torch.float64)
        state = model.initialize_state(mask.shape)
        torch.testing.assert_close(state.tempo, torch.full_like(mask, config.tempo_inhibition_mj_cm2))
        control = model.prepare_control(mask, state.shape)
        state = replace(state, o2=control.energy * 0.5, tempo=control.energy.clone())
        after = model.step_prepared(state, control)
        torch.testing.assert_close(after.tempo, control.energy * 0.5)
        torch.testing.assert_close(after.dose, torch.zeros_like(mask))
        torch.testing.assert_close(after.reaction_progress, torch.zeros_like(mask))


if __name__ == "__main__":
    unittest.main()
