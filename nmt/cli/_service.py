def nmt_service(args):
    """
    :param args:
    :return:
    """
    import os
    from nmt.service.server import HTTPServer

    package_path = args.package_path
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    device = args.device
    workers = args.workers
    port = args.port

    server = HTTPServer(package_path=package_path,
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        batch_size=1,
                        device=device,
                        workers=workers,
                        port=port)

    server.run()
