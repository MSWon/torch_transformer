def nmt_download(args):
    from nmt.service.utils import download_file_from_google_drive
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang

    download_file_from_google_drive(src_lang, tgt_lang)