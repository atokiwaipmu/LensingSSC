from pathlib import Path
import logging
from lensing_ssc.core.patch.noise import add_noise_to_patches

def main():
    """
    ノイズ付加処理を実行するメイン関数。
    指定されたパラメータで収束マップにノイズを追加します。
    """
    logging.basicConfig(level=logging.INFO)
    
    # データディレクトリの設定
    data_dir = '/lustre/work/akira.tokiwa/Projects/LensingSSC/data'
    
    # ノイズ付加処理の実行
    add_noise_to_patches(
        epsilon=0.26,  # シェイプノイズの振幅パラメータ
        ngal_list=[7, 15, 30, 50],  # 銀河密度（銀河数/平方アークミニッツ）のリスト
        data_dir=data_dir,
        overwrite=True  # 既存ファイルを上書きしない
    )
    
    logging.info("ノイズ付加処理が完了しました")

if __name__ == "__main__":
    main()