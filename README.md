# PAE
Projection-augmentation embedding for CLIP-based latent manipulation methods

Please first install requirements by running
```
pip install -r requirements.txt
```

The semantic face editing experiment can be run by
```
python run_models.py [OPTIONS]
```

Please refer to
```
python run_models.py --help
```
for help for passing the arguments.

In particular, the two most important arguments are `--method` and `--target` (when `--target=text`, this is the naive approach of using text embedding as the target). 
