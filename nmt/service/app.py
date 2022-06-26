from fastapi import FastAPI
from nmt.service.utils import TranslateRequest
from nmt.inference import ServiceTransformer


def handle(model: ServiceTransformer):
    async def _handle(params: TranslateRequest):
        src_lang = params.SrcLang
        tgt_lang = params.TgtLang
        text = params.Text

        output_sent = model.infer([text])
        output_data = {
            "SrcLang": src_lang,
            "TgtLang": tgt_lang,
            "InputText": text,
            "TranslatedText": output_sent
        }
        return output_data
    return _handle


def create_app(package_path, src_lang, tgt_lang, batch_size, device):
    model = ServiceTransformer(package_path=package_path,
                               src_lang=src_lang,
                               tgt_lang=tgt_lang,
                               batch_size=batch_size,
                               device=device)

    app = FastAPI()
    app.add_api_route(
        "/nmt",
        handle(model),
        methods=["POST"]
    )

    return app