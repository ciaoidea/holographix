import unittest

import numpy as np

from holo.tnc.afsk import AFSKModem
from holo.tnc.channel import awgn


class TestAFSKTNC(unittest.TestCase):
    def test_afsk_roundtrip_clean(self) -> None:
        rng = np.random.default_rng(0)
        payload = rng.integers(0, 256, size=64, dtype=np.uint8).tobytes()
        modem = AFSKModem()

        samples = modem.encode(payload)
        decoded = modem.decode(samples)

        self.assertEqual(decoded, [payload])

    def test_afsk_roundtrip_noise(self) -> None:
        rng = np.random.default_rng(1)
        payload = rng.integers(0, 256, size=80, dtype=np.uint8).tobytes()
        modem = AFSKModem()

        samples = modem.encode(payload)
        noisy = awgn(samples, 30.0, rng=rng)
        decoded = modem.decode(noisy)

        self.assertEqual(decoded, [payload])

    def test_afsk_multiple_frames(self) -> None:
        rng = np.random.default_rng(2)
        payload_a = rng.integers(0, 256, size=48, dtype=np.uint8).tobytes()
        payload_b = rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
        modem = AFSKModem()

        samples = np.concatenate([modem.encode(payload_a), modem.encode(payload_b)])
        decoded = modem.decode(samples)

        self.assertEqual(decoded, [payload_a, payload_b])


if __name__ == "__main__":
    unittest.main()
