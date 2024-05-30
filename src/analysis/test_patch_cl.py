import unittest
import numpy as np
import healpy as hp

# The function to be applied to each chunk
def run_Clkk_patch(patch, start_idx, end_idx, nside, lmax=5000):
    print(f"Running patch from {start_idx} to {end_idx}, {end_idx - start_idx} elements")
    if len(patch) != end_idx - start_idx:
        raise ValueError(f"Invalid patch size: {len(patch)} != {end_idx - start_idx}")
    if start_idx < 0 or end_idx > hp.nside2npix(nside):
        raise ValueError(f"Invalid indices: {start_idx}, {end_idx}")
    if start_idx >= end_idx:
        raise ValueError(f"Invalid indices: {start_idx}, {end_idx}")

    # Make an array that has the same size as the original map
    patch_full = np.zeros(hp.nside2npix(nside))
    patch_full[start_idx:end_idx] = patch
    patch_full = hp.ma(patch_full)
    patch_full.mask = np.zeros(patch_full.size, dtype=bool)
    patch_full.mask[:start_idx] = True
    patch_full.mask[end_idx:] = True

    patch_full = hp.reorder(patch_full, n2r=True)
    
    cl = hp.anafast(patch_full.filled(), lmax=lmax)
    return cl

class TestRunClkkPatch(unittest.TestCase):

    def setUp(self):
        self.nside = 32  # Smaller nside for testing purposes
        self.lmax = 50  # Smaller lmax for testing purposes
        self.npix = hp.nside2npix(self.nside)
        self.base_pix = 8**2

    def test_random_array(self):
        # Create a random patch array
        patch = np.random.rand(self.npix)

        cls = []
        for i in range(0, self.npix, self.base_pix):
            start_idx = i
            end_idx = i + self.base_pix
            chunk = patch[start_idx:end_idx]

            # Run the function with the random patch
            cl = run_Clkk_patch(chunk, start_idx, end_idx, self.nside, self.lmax)
            cls.append(cl)

        # Assert the output is as expected (not None and has correct length)
        self.assertIsNotNone(cl, "Output should not be None")
        self.assertEqual(len(cl), self.lmax + 1, f"Output length should be {self.lmax + 1}")
        self.assertEqual(len(cls), self.npix // self.base_pix, f"Number of chunks should be {self.npix // self.base_pix}")

        # Additional checks can be added to verify the content of cl if necessary

if __name__ == '__main__':
    unittest.main()
