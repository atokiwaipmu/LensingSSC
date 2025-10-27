from lensing_ssc.core.patch.generator import PatchGenerator
from lensing_ssc.core.patch.processor import PatchProcessor


def main():
    pp = PatchProcessor(patch_size_deg=10)
    zs_list = [0.5, 1.0, 1.5, 2.0, 2.5]
    generator = PatchGenerator(pp, zs_list)
    generator.run()


if __name__ == "__main__":
    main()
