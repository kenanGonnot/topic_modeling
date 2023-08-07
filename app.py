import base64
import json
import os

import numpy as np
import spacy
from flask import Flask, jsonify, request

# from text_summarization import extractive_summarization, abstractive_summarization
from topic_modeling import topic_extraction

spacy.cli.download("en_core_web_sm")
app = Flask(__name__)

error_message = {
    "error_message": "error"
}


@app.route('/v1/inference/topic_modeling', methods=["GET"])
def topic_modeling():
    embedding_model = spacy.load("en_core_web_sm")
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json':
        data = request.get_json()
        data = json.loads(data)
        app.logger.debug(data)
        input_text = str(data['input_text'])

        paragraph_topic_modeled, top_nearest_indices_by_clusters, cluster_image, data_lemmatized, df_results = topic_extraction(
            input_text, embedding_model)  # Topic modeling

        app.logger.debug(top_nearest_indices_by_clusters)
        top_nearest_indices_by_clusters = np.array(top_nearest_indices_by_clusters).tolist()

        encoded_image = base64.b64encode(cluster_image.read()).decode('utf-8')

        response = {
            "cluster_image": encoded_image,
            "df_results": df_results.to_dict()
        }
        return jsonify(response)
    else:
        return jsonify(error_message)


# @app.route('/extractiveSummarization/<string:key_wikidata>', methods=["GET"])
# def extractive_summary(key_wikidata):
#     """
#     Extractive Summarization
#     get paragraph from ovo-mentor-qcm server Flask
#     """
#     content_type = request.headers.get('Content-Type')
#     if content_type == 'application/json':
#         json_paragraph = request.get_json()
#         json_paragraph = json.loads(json_paragraph)
#         print("json_paragraph : {}".format(json_paragraph))
#         paragraphs = json_paragraph['paragraph']
#         print("paragraphs : {}".format(paragraphs))
#         if paragraphs is [] or paragraphs is None:
#             return jsonify("No paragraph found")
#     else:
#         return jsonify("Content-Type must be application/json")
#     summary = extractive_summarization(paragraphs)
#     return summary
#
#
# @app.route('/abstractiveSummarization/<string:key_wikidata>', methods=["GET"])
# def abstractive_summary(key_wikidata):
#     """
#     Abstractive Summarization
#     get paragraph from ovo-mentor-qcm server Flask
#     """
#     content_type = request.headers.get('Content-Type')
#     if content_type == 'application/json':
#         json_paragraph = request.get_json()
#         json_paragraph = json.loads(json_paragraph)
#         paragraphs = json_paragraph['paragraph']
#         if paragraphs is [] or paragraphs is None:
#             return jsonify("No paragraph found")
#     else:
#         paragraphs = request.args.get('paragraph')
#     model = "JulesBelveze/t5-small-headline-generator"
#     # model = "deep-learning-analytics/wikihow-t5-small"
#     # model = "google/pegasus-xsum"
#     # model = "thekenken/text_summarization"
#     framework = "pt"  # pytorch or tf (tensorflow)
#     summarized = abstractive_summarization(paragraphs, model, framework)
#     return summarized


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5003))
    app.run(debug=True, host='0.0.0.0', port=port)
