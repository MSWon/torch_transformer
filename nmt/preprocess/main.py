import argparse

from nmt.preprocess import PreProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", required=True)

    args = parser.parse_args()

    processor = PreProcessor(args.config_path)
    processor.execute()