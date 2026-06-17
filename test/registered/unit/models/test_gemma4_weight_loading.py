"""Unit tests for the Gemma4 weight-loading completeness check -- CPU only.

The unloaded-parameter diagnostic lives in ``verify_weights_loaded``, which the loader
runs only on a full load; ``load_weights`` itself no longer emits it (online updates
call ``load_weights`` directly and skip the check).
"""

import unittest

import torch

from sglang.srt.models.gemma4_causal import Gemma4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGemma4WeightLoading(unittest.TestCase):
    def _make_minimal_causal_model(self):
        model = object.__new__(Gemma4ForCausalLM)
        model.named_parameters = lambda: iter([("missing.weight", torch.empty(1))])
        model.named_buffers = lambda: iter([])
        model.named_modules = lambda: iter([])
        return model

    def test_verify_warns_on_unloaded_parameters(self):
        model = self._make_minimal_causal_model()
        with self.assertLogs("sglang.srt.models.gemma4_causal", level="INFO") as logs:
            model.verify_weights_loaded(set())
        output = "\n".join(logs.output)
        self.assertIn("Some weights are not initialized from checkpoints", output)
        self.assertIn("missing.weight", output)

    def test_verify_is_silent_when_all_loaded(self):
        model = self._make_minimal_causal_model()
        with self.assertNoLogs("sglang.srt.models.gemma4_causal", level="INFO"):
            model.verify_weights_loaded({"missing.weight"})


if __name__ == "__main__":
    unittest.main()
