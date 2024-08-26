### FQC toolkit demo

1. **Override**: reading `user_path.json` and synchronize with the default configuration.
2. **Dataset**: loading the dataset from the address given by `user_path.json`
3. **Sanitizing Data**: contains removing duplicates, pruning dataset, removing null data.
4. **Feature Cleaning**: including dropping some features, and removing constant features.
5. **Adding Class**: calculate and assign class regarding some features.
6. **Data Statistics**: describing data and present data in a small size.
7. **Feature Relation**: report data size, and feature relationship.
8. **Train a Model**: including preparing data to train a preliminary model..
9. **Performance Metric**: reporting performance of model on a dataset in terms of accuracy, precision, recall, f1-score, support.
10. **Feature Importance**: using the trained model, to compute feature importance in three different methods.
11. **Report Feature Importance**: reporting feature importance in the three different methods.


### How to modify user configuration:
```sh
vim user_path.json
```

### How to run the script:
```sh
python3 fqc.py --user-path user_path &>demo.txt
```