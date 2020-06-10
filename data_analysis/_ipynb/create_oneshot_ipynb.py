"""One shot script
Generate ipynb coloring for each sheet for each colum in excel
==> screenshot for presentation
"""

import json
import pandas as pd
import re
import os

cell_list = []

# Adding libraries
cell_list.append({
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import sys, os\n",
        "import ast\n",
        "import re\n",
        "import seaborn as sns\n",
        "import inspect"
    ]
})

# Adding data analysis
# coloration: https://kanoki.org/2019/01/02/pandas-trick-for-the-day-color-code-columns-rows-cells-of-dataframe/
cell_list.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def color_with_value(val):\n",
        "    color = 'green' if val < 0.001 else ('orange' if val < 0.01 else 'red')\n",
        "    return 'background-color: ' + color"
    ]
})

# loading data to now which sheets
currdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
xl = pd.ExcelFile(os.path.join(currdir,'data/pvalues.xlsx'))
features = list(xl.sheet_names)

print(features)
for feat in features:
    # Adding name
    cell_list.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# "+feat
    ]
    })
    # Adding data
    cell_list.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "CURRENTDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))\n",
            "results = pd.read_excel(os.path.join(CURRENTDIR,\"data/pvalues.xlsx\"), sheet_name='"+feat+"', index_col=0)"
        ]
    })
    for col in ["Agent[T.R]", feat, feat+":Agent[T.R]"]:
        cell_list.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df = results[['"+col+"']].sort_values(by='"+col+"')\n",
                "df[df['"+col+"'] < 0.05].style.applymap(color_with_value)"
            ]
            })

data = {
    "cells":cell_list,
    "metadata": {
        "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.7.6-final"
    },
    "orig_nbformat": 2,
    "kernelspec": {
        "name": "python37364bitanaconda3virtualenvf4ae0adae11f4651a06ea79eb5fcd3ff",
        "display_name": "Python 3.7.3 64-bit ('anaconda3': virtualenv)"
    }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(os.path.join(currdir,'data_analysis/_ipynb/oneshot_ipynb.ipynb'), 'w') as json_file:
    json.dump(data, json_file)

"""
# Later analysis
thres = [0.001, 0.01, 0.05, 1]
table = []
warnings = {}
for file in ['data/pvalues.xlsx','data/pvalues_align.xlsx']:
    xl = pd.ExcelFile(file)
    features = list(xl.sheet_names)
    for feat in features:
        results = pd.read_excel(file, sheet_name = feat, index_col=0)
        print(feat, results.columns)
        for col in [c for c in results.columns if c not in ['Intercept', 'Warning']]:
            for i, t in enumerate(thres):
                store = {}
                store['feature'] = '_'.join(feat.split('_')[:-1])
                store['condition'] = feat.split('_')[-1]
                if col == 'Agent[T.R]':
                    store['fixed_effect'] = col
                else:
                    store['fixed_effect'] = '_'.join(col.split('_')[:-1]) if (re.search(':', col) is None) else 'interaction'
                store['threshold'] = t
                if i == 0:
                    store['areas'] = str(list(results[results[col] <= t].index))
                else:
                    store['areas'] = str(list(results[(results[col] <= t) & (results[col] > thres[i-1])].index))
                table.append(store)
        if store['feature'] in warnings.keys():
            warnings[store['feature']][store['condition']] = results['Warning'].apply(lambda x: 1 if (str(x).lower() != 'nan') else 0).sum()/results.shape[0]
        else:
            warnings[store['feature']] = {store['condition'] : results['Warning'].apply(lambda x: 1 if (str(x).lower() != 'nan') else 0).sum()/results.shape[0]}

pd.DataFrame(table).to_excel('data_neuro/neuro_summary.xlsx', index=False)
pd.DataFrame(warnings).to_excel('data_neuro/neuro_errors.xlsx')
"""