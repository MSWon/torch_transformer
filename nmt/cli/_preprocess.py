def nmt_preprocess(args):
    from nmt.preprocess import PreProcessor

    processor = PreProcessor(args.config_path)
    processor.execute()