from flask import Flask, request, jsonify, render_template
import automl
import json

app = Flask(__name__)

@app.route('/')
def info():
  return render_template('info.html')



@app.route('/get_data', methods=['GET', 'POST'])
def get_data():
  dataset_name = request.values.get('dataset', None)
  if dataset_name is None:
    print('-- No dataset name given --')
    response = jsonify({ 'success': False, 'message': 'No dataset given' })
  else:
    NBR_OF_ROWS = 10
    df, target_name = automl.get_data(dataset_name, NBR_OF_ROWS)
    response = jsonify({
      'dataset': df.to_dict(),
      'target': target_name
    })
  response.headers.add('Access-Control-Allow-Origin', '*')
  return response

  

@app.route('/predict', methods=['GET', 'POST'])
def predict():
  dataset_name = request.values.get('dataset', None)
  print(f'received data set name {dataset_name}')
  if dataset_name is None:
    print('-- No dataset name given --')
    return jsonify({ 'success': False, 'message': 'No dataset given' })
  response = automl.predict_dataset(dataset_name)
  # response = jsonify(json.loads('''
  # {
  #   "error_mae": 0.29214366588893054,
  #   "error_mape": 0.1578641635085599,
  #   "models_board": {
  #       "cost": {
  #           "16": 0.18051776152542465,
  #           "20": 0.17095374223723914,
  #           "22": 0.1745115791465276,
  #           "30": 0.18500712150285392
  #       },
  #       "duration": {
  #           "16": 6.800856828689575,
  #           "20": 7.258382081985474,
  #           "22": 3.5819990634918213,
  #           "30": 3.2903990745544434
  #       },
  #       "ensemble_weight": {
  #           "16": 0.25,
  #           "20": 0.25,
  #           "22": 0.25,
  #           "30": 0.25
  #       },
  #       "rank": {
  #           "16": 3,
  #           "20": 1,
  #           "22": 2,
  #           "30": 4
  #       },
  #       "type": {
  #           "16": "gradient_boosting",
  #           "20": "gradient_boosting",
  #           "22": "gradient_boosting",
  #           "30": "gradient_boosting"
  #       }
  #   },
  #   "predictions": {
  #       "actual": {
  #           "0": 5.00001,
  #           "1": 1.88,
  #           "2": 1.583,
  #           "3": 0.705
  #       },
  #       "predicted": {
  #               "0": 3.331787884235382,
  #               "1": 1.848462074995041,
  #               "2": 1.5753859877586365,
  #               "3": 0.7649817168712616
  #       }
  #   }
  # }'''))

  response.headers.add('Access-Control-Allow-Origin', '*')
  return response


# @app.route('/automl')
# def automl():
#   return True


if __name__ == '__main__':
  app.run()
