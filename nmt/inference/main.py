import argparse

from nmt.inference import ServiceTransformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package_path", required=True)
    parser.add_argument("--src_lang", required=True)
    parser.add_argument("--tgt_lang", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--device", required=False, default="cpu")

    args = parser.parse_args()

    model = ServiceTransformer(package_path=args.package_path,
                               src_lang=args.src_lang,
                               tgt_lang=args.tgt_lang,
                               batch_size=args.batch_size,
                               device=args.device)

    model.infer_corpus(args.corpus_path)