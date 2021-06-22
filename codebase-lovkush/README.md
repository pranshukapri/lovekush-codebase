# Brief overview of directories
* dash_app: contains app.py file to create plotly dash app
* data_preprocessed: folder where "preprocessed" data is stored. The only difference between data_raw and data_preprocessed is that some duplicate files or bad files have been removed. (So probably not the best name for this directory)
* data_preprocessed_csv: this is where preprocessed data is stored. The answers and questions from the text files are put into a dataframe, and then stored in a csv file.
* data_raw: folder where raw .txt files of deposition transcripts are stored
* eda: this contains the analysis that I did
    * eda_depositions.ipynb: given the 150 or so depositions, could we cluster them and also find keywords for them.
    * eda_lawyer_stats.ipynb: 'profiling' the lawyers. e.g. average number of words per laywer.
    * edq_q_clusters.ipynb: given a case, create a visual that clusters all the qeustions. this was the first example in my video
    * eda_sentiment.ipynb: a basic attempt at using vader for sentiment analysis. basically no useful signal
    * lawyers_stats.csv: csv file containing stats from eda_lawyer_stats
    * words_to_remove.txt: file containing list of words that I considered worth adding to list of stop words; primarily used in eda_depositions to help identify good keywords
* pickled vectors: in eda_q_clusters, the main class has ability to save most of its state in a .pkl file, and can then load the state back.
* preprocessing: this folder contains scripts to convert raw data into cleaner data
    * deleted_bad.txt: list of filenames that are deleted because my scripts determine they are not actually a deposition transcript
    * deleted_duplicates: list of filenames that were deleted because they were duplicates of another file
    * deleted_manual: filenames that I manually deleted because they were not depositions
    * metadata.csv: a csv file containing metadata that I manually created
    * metadata.ipynb: file where you can see code I used to manually create metadata.csv
    * preprocessing.ipyng: notebook where you can see code I used to do the cleaning/preprocessing
