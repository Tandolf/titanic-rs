# Titanic Survivors

This is my attempt of using a random forest algorithm on some training data supplied from kaggle.com. The purpose of
this exercise was to try to estimate how many survived vs perished on the MS Titanic.

The data folder contains training data for the ML algorithm, and then we run the classifier on the rest of the passanger
list to estimate the correct outcome.

Written in rust, using the crates `polars` for handling the data, and `smartcore` for running the algorithms.
