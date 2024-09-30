
from src.kappa_smoother import KappaSmoother
from src.utils import filter_config, load_config, parse_arguments

def main():
    args = parse_arguments()
    config = load_config(args.config_file)
    filtered_config = filter_config(config, KappaSmoother)

    smoother = KappaSmoother(datadir=args.datadir, **filtered_config, overwrite=args.overwrite)
    smoother.smooth_kappa()

if __name__ == "__main__":
    main()