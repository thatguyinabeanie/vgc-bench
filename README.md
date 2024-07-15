# How to setup
Prerequisites:
1. Python
1. NodeJS and npm (to run the showdown server locally)
1. Git

The newest of each should be fine.

Run the following (assuming Linux):
```
git clone git@github.com:cameronangliss/UT-masters-thesis.git
cd UT-masters-thesis
git submodules update --init
cd pokemon-showdown
node pokemon-showdown start --no-security
```
Let that run until you see the following text:
```
RESTORE CHATROOM: lobby
RESTORE CHATROOM: staff
Worker 1 now listening on 0.0.0.0:8000
Test your server at http://localhost:8000
```
Then Ctrl+c the operation and run the following:
```
cd ..
python3 -m venv <env-path>
source <env-path>/bin/activate
pip install -r requirements.txt
```

# How to run
1. Navigate to project root (`UT-masters-thesis/`)
1. Ensure that the Python virtual environment is activated (see above setup script for how to do this)
1. Run `python src/train.py` to train, or `python src/play.py` to play on the online servers. Run `python src/play.py --help` for option details.
