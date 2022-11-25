# automl sandbox



https://user-images.githubusercontent.com/5780544/204021439-c0106549-6f20-4743-ba84-c9b803f6cce8.mov



## About

Created during a 2-day Hackathon mainly for the purpose of learning more about currently available Auto ML techniques. It also serves as a proof of concept of how easily Auto ML libraries like autosklearn and autokeras (etc.) can exposed through a web interface benefitting everyone, from data scientists and analysts to those with less knowledge about machine learning but perhaps with more domain knowledge.

## Install backend

```sh
# auto-sklearn failed for me on python=3.10 so ensure you have python=3.9
conda create -n "automl39" python=3.9
conda activate automl39
```

### Option A. Install automatically from our requirements.txt
```sh
pip install -r requirements.txt
```
Or, if that doesn't work, use:
```sh
pip install -r requirements_all.txt
```

### Option B. Install "manually"
```sh
# Install auto-sklearn dependencies from a remote requirements file
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install

# Finally install auto-sklearn
pip install auto-sklearn
```

## Install client (for interacting with the backend)

Assuming you have `node` and `npm` installed:
```sh
cd ./src/web
npm i
```


## Run it

Backend:
```sh
# Terminal tab 1
cd ./src/web
npm run start:backend
```

Client:
```sh
# Terminal tab 1
cd ./src/web
npm run start:client
```
