"""Unit tests for the loader's load-and-verify helper -- CPU only, no GPU required.

The completeness check lives in ``verify_weights_loaded``, which the loader runs only
after a full load. Models that return ``None`` or define no hook are left untouched
(online weight-update paths call ``load_weights`` directly and skip the check).
"""

import unittest

from sglang.srt.model_loader.loader import _load_and_verify_weights
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ModelWithCheck:
    def __init__(self):
        self.verified = None

    def load_weights(self, weights):
        return {"w1", "w2"}

    def verify_weights_loaded(self, loaded_params):
        self.verified = loaded_params


class _ModelReturningNone:
    def __init__(self):
        self.verified = False

    def load_weights(self, weights):
        return None

    def verify_weights_loaded(self, loaded_params):
        self.verified = True


class _ModelWithoutCheck:
    def load_weights(self, weights):
        return {"w1"}


class TestLoadAndVerifyWeights(unittest.TestCase):
    def test_runs_check_after_full_load(self):
        model = _ModelWithCheck()
        result = _load_and_verify_weights(model, [])
        self.assertEqual(result, {"w1", "w2"})
        self.assertEqual(model.verified, {"w1", "w2"})

    def test_skips_check_when_load_returns_none(self):
        model = _ModelReturningNone()
        _load_and_verify_weights(model, [])
        self.assertFalse(model.verified)

    def test_model_without_check_hook_is_left_untouched(self):
        result = _load_and_verify_weights(_ModelWithoutCheck(), [])
        self.assertEqual(result, {"w1"})


if __name__ == "__main__":
    unittest.main()
