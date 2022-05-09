from flask import Flask, request, jsonify
#from flask_classful import FlaskView, route
from nmt.inference import ServiceTransformer
import yaml
import argparse
import json


app = Flask(__name__)

@app.route('/nmt', methods=['POST'])
def handle_request():
    params = json.loads(request.get_data(), encoding='utf-8')
    src_lang = params["SrcLang"]
    tgt_lang = params["TgtLang"]
    text = params["Text"]

    output_sent = model.infer([text])
    output_data = jsonify({
        "SrcLang": src_lang,
        "TgtLang": tgt_lang,
        "InputText": text,
        "TranslatedText": output_sent
    })
    return output_data

def create_app(package_path, src_lang, tgt_lang, device="cpu"):
    global model

    model = ServiceTransformer(package_path=package_path,
                               src_lang=src_lang,
                               tgt_lang=tgt_lang,
                               batch_size=1,
                               device=device)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("package_path", help="package path")
    parser.add_argument("--src_lang", "-s", required=True, help="src lang")
    parser.add_argument("--tgt_lang", "-t", required=True, help="tgt lang")

    parser.add_argument("--device", required=False, default="cpu")

    args = parser.parse_args()

    model = ServiceTransformer(package_path=args.package_path,
                               src_lang=args.src_lang,
                               tgt_lang=args.tgt_lang,
                               device=args.device)
    app.run()