from fastapi import FastAPI
from uvicorn import Server, Config
from nmt.inference import ServiceTransformer
from nmt.service.utils import TranslateRequest, check_requested_langpair_support


class HTTPServer(object):
    def __init__(self, 
                 package_path: str,
                 src_lang: str,
                 tgt_lang: str,
                 batch_size: int,
                 device: str,
                 port: int):

        check_requested_langpair_support(src_lang=src_lang,
                                         tgt_lang=tgt_lang)

        self.package_path = package_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self.device = device

        self.model = self.build_model()

        app = FastAPI()
        app.add_api_route(
            "/nmt",
            self.handle(),
            methods=["POST"]
        )

        config = Config(app=app, port=port, host="0.0.0.0")
        self.server = Server(config)


    def build_model(self):
        model = ServiceTransformer(package_path=self.package_path,
                                   src_lang=self.src_lang,
                                   tgt_lang=self.tgt_lang,
                                   batch_size=self.batch_size,
                                   device=self.device)
        return model


    def run(self):
        return self.server.run()


    def handle(self):
        async def _handle(params: TranslateRequest):
            src_lang = params.SrcLang
            tgt_lang = params.TgtLang
            text = params.Text

            output_sent = self.model.infer([text])
            output_data = {
                "SrcLang": src_lang,
                "TgtLang": tgt_lang,
                "InputText": text,
                "TranslatedText": output_sent
            }
            return output_data

        return _handle
            

    