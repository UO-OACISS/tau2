# To run this example, you'll need to install plotly, jupyterlab and pandas:

# $ python3 -m pip install -r requirements.txt

# Then add jupyter to your path (for example):

PATH=${HOME}/Library/Python/3.7/bin:${PATH}

# Then add the TAU utilities to your python path (for example):

export PYTHONPATH=${HOME}/src/tau2/apple/python:${PYTHONPATH}

# get the notebook file from argv
if [ "$#" -ne 1 ]; then
  echo "usage: go.sh <jupyter notebook file>";
  exit 1;
fi

# Then run jupyter lab with the jupyter notebook file:

jupyter lab $1

#In your browser that is launched, open the notebook and go!
