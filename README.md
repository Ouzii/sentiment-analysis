# _Cross-lingual Sentiment Preservation in Binary and Multi-dimensional Classification_

The text data used in this project is from:

Lison, P. and Tiedemann, J. (2016), OpenSubtitles2016: Extracting large parallel corpora from movie and tv subtitles, _in_ 'Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016)'.

The code is implemented in Python 3.x using Scikit-learn:

Pedregosa et al. (2011), Scikit-learn: Machine Learning in Python, _in_ Journal of Machine Learning Research 12, pp. 2825-2830.


To run a prediction on csv files:
1. Clone the repository
2. Insert all of your csv files to input-folder
3. Run ./run_analysis.sh $language $dimension $column $as_strings
    where
    - $language: language code for one of the supported languages
    - $dimension: bin/multi
    - $column: Index of the column in which the data resides in csv files. Starts from 0.
    - $as_strings: Define if the output should be numerical or string values.
4. Outputs of the prediction will be written in output-folder

Note: it is assumed that csv input files have a header row as the first row
