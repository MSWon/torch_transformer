def nmt_cmd(args):
    """
    :param args:
    :return:
    """
    from nmt.inference import ServiceTransformer

    ## Build model
    model = ServiceTransformer(package_path=args.package_path,
                               src_lang=args.src_lang,
                               tgt_lang=args.tgt_lang,
                               device=args.device,
                               batch_size=1,
                               log_level="debug")
    ## Infer model
    model.cmd_infer()