
Once in the scitas clusters, one must setup the virtual environment.
```
virtualenv --system-site-packages <venvs_name>
source <venvs_name>/bin/activate
pip install ultralytics --user
pip install thop --user
```

Then a module is required to install
```
module load gcc/8.4.0-cuda python/3.7.7		
```

script expect 8 cores, 4 get a warning