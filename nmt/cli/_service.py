def nmt_service(args):
    """
    :param args:
    :return:
    """
    import os
    from nmt.service.utils import check_requested_langpair_support
    package_path = args.package_path
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    check_requested_langpair_support(src_lang, tgt_lang)

    device = args.device
    port_num = args.port

    fn_args = f'\"{package_path}\", \"{src_lang}\", \"{tgt_lang}\", \"{device}\"'

    cmd = f"gunicorn 'nmt.service.api:create_app({fn_args})' -b 0.0.0.0:{port_num}"
    os.system(cmd)