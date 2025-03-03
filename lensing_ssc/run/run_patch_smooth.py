from pathlib import Path
from lensing_ssc.core.patch.processor import PatchProcessor
from lensing_ssc.core.patch.smoother import PatchSmoother
import logging

def main():
    logging.basicConfig(level=logging.INFO)

    # パッチプロセッサーの初期化（パッチサイズ10度）
    pp = PatchProcessor(patch_size_deg=10.0, xsize=2048)

    # データディレクトリの設定
    data_dir = '/lustre/work/akira.tokiwa/Projects/LensingSSC/data'

    # 平滑化処理の実行
    smoother = PatchSmoother(
        data_dir=data_dir,
        patch_processor=pp,
        sl_list=[2, 5, 8, 10],  # 平滑化の長さ（アークミニッツ）
        overwrite=False
    )
    smoother.run()

if __name__ == "__main__":
    main()