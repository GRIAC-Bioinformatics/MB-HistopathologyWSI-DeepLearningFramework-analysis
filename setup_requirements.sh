pip install virtualenv
python -m venv venv_test
source venv_test/bin/activate

pip install jupyter
pip install ipython
pip install ipykernel
ipython kernel install --user --name=my-venv
python -m ipykernel install --user --name=my-venv
pip install bash_kernel
python -m bash_kernel.install

# Install the needed requirements
pip install -r /workspace/ImageRecognition/requirements.txt