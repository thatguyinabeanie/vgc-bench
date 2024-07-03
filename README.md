# How to setup
Prerequisites:
1. Python
1. NodeJS and npm (to run the showdown server locally)
1. Git

The newest of each should be fine.

Run the following:
```
git clone git@github.com:cameronangliss/UT-masters-thesis.git
cd UT-masters-thesis
git submodules update --init
python3 -m venv <env-path>

# Activate Python virtual environment
# for Linux/Mac:
source <env-path>/bin/activate
# for Windows:
.\<env-path>\Scripts\activate

pip install -r requirements.txt
```

# How to run
In one terminal, navigate to `UT-masters-thesis` and run:
```
cd pokemon-showdown
node pokemon-showdown start --no-security
```
On the first time running this, it will take a while, but that's okay. You should eventually see this:
```
RESTORE CHATROOM: lobby
RESTORE CHATROOM: staff
Worker 1 now listening on 0.0.0.0:8000
Test your server at http://localhost:8000
```
This indicates that you have successfully started hosting Pokemon Showdown locally.

Now do the following:
1. Open a new terminal
1. Navigate to `UT-masters-thesis/`
1. Ensure that the Python virtual environment is activated (see above setup script for how to do this)
1. Run `python src/train.py`.
