# Spam Filter
Building spam filter with the Naive Bayes algorithm.

In this project i'm going to build spam filter directed at preventing mobile phone spam SMS messages, the algorithm is based on [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem).

# Overview of the Naive Bayes algorithm
The main idea is that computer learns from classifications a humans does, and then uses that knowledge to classify new messages.The algorithm analyzes new messages and determines whether they are spam or not. More specifically, it uses that human knowledge to estimate probabilities for new messages — probabilities for spam and non-spam.

The algorithm tries to calculate the following probabilities:
- __P(Spam|New SMS)-?__
- __P(Non-spam|New SMS)-?__

According Bayes' theorem, we have the following:
```math
P(Spam|New SMS)= P(Spam)*P(New SMS|Spam)/P(New SMS)
P(Non-spam|New SMS)= P(Non-spam)*P(New SMS|Non-spam)/P(New SMS)
```
If `P(Spam|New SMS)>P(Non-spam|New SMS)` then computer will classify the new message as spam.

I'll apply the algorithm to a [dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) of over 5,000 SMS messages. This is a SMS Spam Collection Data Set from [Maching learning repository](https://archive.ics.uci.edu/ml/index.php). The collection is a text file, where each line has the correct class followed by the raw message.
