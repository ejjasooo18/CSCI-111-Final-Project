1. First install the required dependencies using:

    pip install -r requirements.txt

2. Double check the dataset being used on line 11 of preprocessing.py

    Currently hardcoded to winequality-red.csv, the dataset used for this problem

3. Run the code

    python preprocessing.py

    Will generate a heatmap of the different wine components of our dataset,
    allowing the user to see which components contribute the most to wine quality.

    Will then show a histogram showing the different features of the dataset, to
    better visualize the type of data being used to train the two models.

    Confusion matrix is generated at the end of the program, report for precision, recall, 
    f1-score and support is printed onto the console at the end as well.

    Note: Current graph must be closed to show the following ones.