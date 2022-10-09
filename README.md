# Personalized Query Refinement

This project explores personalizing the changes to the query of a user (refinement) in order to increase the quality of retrieved documents and hence, search experience.

## Installation
You need to have ``Python=3.8`` and install [`pyserini`](https://github.com/castorini/pyserini/) package, among others listed in [``requirements.txt``](requirements.txt):

By ``pip``, clone the codebase and install the required packages:
```sh
git clone https://github.com/Fani-Lab/personalized_query_refinement
cd personalized_query_refinement
pip install -r requirements.txt
```

By [``conda``](https://www.anaconda.com/products/individual):

```sh
git clone https://github.com/Fani-Lab/personalized_query_refinement
cd personalized_query_refinement
conda env create -f environment.yml
conda activate pqr
```

**Note:** when installing jdk11, remember to check your env path for the executable in windows OS.

