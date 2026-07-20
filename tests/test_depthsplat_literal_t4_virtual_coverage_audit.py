"""CPU contracts for literal T=4 source-only virtual coverage diagnostics."""

from __future__ import annotations

import inspect
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _virtuals():
    means = torch.tensor(
        [(0.1 if index < 6 else 0.7, 0.0, 2.0) for index in range(12)],
        dtype=torch.float64,
    )
    covariances = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3).repeat(12, 1, 1)
    covariances = covariances * 0.01
    opacities = torch.full((12,), 0.5, dtype=torch.float64)
    return means, covariances, opacities


def _audit(candidate_scale: float):
    from saes.depthsplat_literal_t4_virtual_coverage_audit import (
        audit_literal_virtual_anchor_set,
    )

    means, covariances, opacities = _virtuals()
    return audit_literal_virtual_anchor_set(
        virtual_means=means,
        virtual_covariances=covariances,
        virtual_opacities=opacities,
        merged_mean=torch.tensor((0.0, 0.0, 2.0), dtype=torch.float64),
        merged_covariance=torch.eye(3, dtype=torch.float64) * 0.01 * candidate_scale,
        merged_opacity=torch.tensor(0.5, dtype=torch.float64),
        context_extrinsic=torch.eye(4, dtype=torch.float64),
        context_intrinsic=torch.eye(3, dtype=torch.float64),
    )


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class LiteralT4VirtualCoverageAuditTest(unittest.TestCase):
    def test_fixed_scale_anchor_audit_reports_holes_and_strict_continuity_break(self):
        result = _audit(candidate_scale=1.0)

        self.assertAlmostEqual(result["fixed_moment_covariance_scale"], 1.0)
        self.assertEqual(result["virtual_primitive_count"], 12)
        self.assertIs(result["coverage"]["valid"], True)
        self.assertGreater(result["coverage"]["hole_count"], 0)
        self.assertIs(result["continuity"]["checked"], True)
        self.assertIs(
            result["continuity"]["all_active_virtual_2sigma_supports_contained"],
            False,
        )
        self.assertIs(result["continuity"]["broken"], True)

    def test_larger_candidate_support_closes_the_same_virtual_set(self):
        result = _audit(candidate_scale=64.0)

        self.assertIs(result["coverage"]["valid"], True)
        self.assertEqual(result["coverage"]["hole_count"], 0)
        self.assertAlmostEqual(result["coverage"]["count_recall"], 1.0)
        self.assertAlmostEqual(result["coverage"]["mass_weighted_recall"], 1.0)
        self.assertIs(
            result["continuity"]["all_active_virtual_2sigma_supports_contained"],
            True,
        )
        self.assertIs(result["continuity"]["broken"], False)

    def test_public_apis_and_cli_expose_no_target_or_v1_default(self):
        from saes.depthsplat_literal_t4_virtual_coverage_audit import (
            audit_literal_t4_virtual_anchor_coverage,
            audit_literal_virtual_anchor_set,
        )
        from scripts.saes_depthsplat_literal_t4_virtual_coverage_audit import (
            DEFAULT_INPUT_ROOT,
            build_parser,
        )

        for function in (
            audit_literal_virtual_anchor_set,
            audit_literal_t4_virtual_anchor_coverage,
        ):
            self.assertTrue(
                all("target" not in name for name in inspect.signature(function).parameters)
            )

        parser = build_parser()
        args = parser.parse_args(
            [
                "--output-dir",
                "result",
                "--literal-t4-v16-record",
                "frozen/v16t4.json",
            ]
        )
        self.assertEqual(args.input_root, DEFAULT_INPUT_ROOT)
        self.assertEqual(DEFAULT_INPUT_ROOT.name, "depthsplat_sample0_l0_l1_context_only_v2")
        self.assertFalse(hasattr(args, "target_index"))
        self.assertFalse(hasattr(args, "target_indices"))
        self.assertFalse(hasattr(args, "sample_index"))


if __name__ == "__main__":
    unittest.main()
