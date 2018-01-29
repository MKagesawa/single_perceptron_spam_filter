# single_perceptron_spam_filter
Supervised Learning: Pre-processed input spam filter using perceptron algorithm

## Note: Pre-processed data
- First column of the text files indicate the label. 1 = spam, 0 = not spam.
- This program does not take raw email as input.

The following have already been implemented to the data sets:
1. Lower-casing, removal of HTML tags, normalization of URLs, e-mail addresses, and numbers.
2. Words are reduced to their stemmed form. e.g. “discount”, “discounts”, “discounted”, “discounting” -> “discount”


This email for example:
> Anyone knows how much it costs to host a web portal?
> Well, it depends on how many visitors youre expecting. This can be anywhere from
>less than 10 bucks a month to a couple of $100. You should checkout
>http://www.rackspace.com/ or perhaps Amazon EC2 if youre running something big..
>To unsubscribe yourself from this mailing list, send an email to: groupnameunsubscribe@
>egroups.com

Becomes:
>anyon know how much it cost to host a web portal well it depend on how mani visitor
>your expect thi can be anywher from less than number buck a month to a coupl of
>dollarnumb you should checkout httpaddr or perhap amazon ecnumb if your run
>someth big to unsubscrib yourself from thi mail list send an email to emailaddr

## Process
1. file_prep.py takes in pre-processed train and test data, and separate spam/not spam label
2. vector.py: words(lines) function makes a list of frequently occuring words from train data. Default set to words that appear in more than 25 emails.
3. vector.py: feature_vector() fuction creates a list in list indicating weather the fretly occuring words is in each email.
4. train.py: perceptron_train() function finds the weight parameters of the model.
```
# The training algorithm runs until there is no error classifying the training set or reaching max iteration user has set
    while (perceptron_error(w, vector, label) != 0) and (iteration < max_iter):
        iteration += 1
        for i in range(max_pass):
            result = np.dot(w, matrix[i])
            # Correctly classified
            if result * label[i] > 0:
                w = w
            else:
                k += 1
                w += np.multiply(label[i], matrix[i])
```
5. train.py: perceptron_error(w, vector, label) function takes the weight parameters of a model and tests it against validation or test data sets
6. train.py: most_positive_negative(w, freq_words) function finds the words most indicative of spam or not spam (parameters with highest or lowest weight)
7. train.py: plot() function plots validation error & number of iterations with different input sizes N.

## Acknowledgements
*Inspired by problem sets developed by Keith Ross, David Sontag, Andrew Ng.

*Dataset used comes from SpamAssassin Public Corpus.
