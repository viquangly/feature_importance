
Hello there!

If you're like me, and work in the finance / banking industry,
then you'll understand the rigor of going through ***model validation***. 
***Feature selection*** is one of the key sections in every model documentation.
Hopefully, this code will provide a generalizable way to identify which features to keep.

Use the `RecursiveFeatureSelection` class to iteratively remove less important features with each iteration.
Then, use the `Visualizers` to show the model performance as more and more features are removed.

Now, you may ask: "Why don't you just use `sklearn.feature_selection.RFECV`?"  
- `RFECV` provides less transparency
as it only gives you one model in the end with the optimal number of features.  However, you may want more
freedom to choose for yourself. 
-  Most model validation departments will want you to "show" your work.

Currently, this only supports ***binary classification***, as that is the bulk of my models.
