# Spam Filter
Building spam filter with the Naive Bayes algorithm.

In this project we are going to build spam filter directed at preventing mobile phone spam SMS messages, the algorithm is based on [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem).

# Overview of the Naive Bayes algorithm
The main idea is that computer learns from classifications a humans does, and then uses that knowledge to classify new messages. The algorithm analyzes new messages and determines whether they are spam or not. More specifically, it uses that human knowledge to estimate probabilities for new messages — probabilities for spam and non-spam.

The algorithm tries to calculate the following probabilities:
- __P(Spam|New SMS)-?__
- __P(Non-spam|New SMS)-?__

Or in other words, tries to answer the questions:
- __What's the probability that this new SMS message is spam, given its content?__
- __What's the probability that this new SMS message is non-spam, given its content?__

According to Bayes' theorem, we have the following:

<img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20P%28Spam%5Cmid%20New%20SMS%29%20%3D%20%5Cfrac%7BP%28Spam%29%5Ctimes%20P%28New%20SMS%5Cmid%20Spam%29%7D%7BP%28New%20SMS%29%7D">
<img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20P%28Non-spam%5Cmid%20New%20SMS%29%20%3D%20%5Cfrac%7BP%28Non-spam%29%5Ctimes%20P%28New%20SMS%5Cmid%20Non-spam%29%7D%7BP%28New%20SMS%29%7D">

As the main goal of the algorithm is to classify new messages, not to calculate probabilities, we safely can ignore division by __P(New SMS)__. So it's more accurate to replace the equality sign with ∝ (directly proportional) in equations:

<img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20P%28Spam%5Cmid%20New%20SMS%29%20%5Cpropto%20P%28Spam%29%5Ctimes%20P%28New%20SMS%5Cmid%20Spam%29">
<img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20P%28Non-spam%5Cmid%20New%20SMS%29%20%5Cpropto%20P%28Non-spam%29%5Ctimes%20P%28New%20SMS%5Cmid%20Non-spam%29">

Since we treat each word separately, and we assume conditional independence between them (it's unrealistic in real world, that's why algorithm is called __naive__), therefore in general if new SMS message has __n__ words (w<sub>i</sub>), we can simplify equations to this form:

<img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20P%28Spam%5Cmid%20w_%7B1%7D%2C%20w_%7B2%7D%2C...w_%7Bn%7D%29%20%5Cpropto%20P%28Spam%29%5Ctimes%20P%28w_%7B1%7D%5Cmid%20Spam%29%5Ctimes%20P%28w_%7B2%7D%5Cmid%20Spam%29%5Ctimes%20...%5Ctimes%20P%28w_%7Bn%7D%5Cmid%20Spam%29">

<img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20P%28Non-spam%5Cmid%20w_%7B1%7D%2Cw_%7B2%7D%2C...w_%7Bn%7D%29%20%5Cpropto%20P%28Non-spam%29%5Ctimes%20P%28w_%7B1%7D%5Cmid%20Non-spam%29%5Ctimes%20P%28w_%7B2%7D%5Cmid%20Non-spam%29%5Ctimes...%5Ctimes%20P%28w_%7Bn%7D%5Cmid%20Non-spam%29">

But if we receive a new message that contains words which are not part of the vocabulary, we will get zero probabilities in equations. And as result turning the entire equation to 0. To fix the problem, we're going to add a [smoothing parameter](https://en.wikipedia.org/wiki/Additive_smoothing) __α__ for every word.
In this way, we have:

<img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20P%28w_%7Bi%7D%5Cmid%20Spam%29%20%3D%20%5Cfrac%7BN_%7Bw_%7Bi%7D%5Cmid%20Spam%7D%20&plus;%20%5Calpha%20%7D%7BN_%7BSpam%7D%20&plus;%20%5Calpha%20%5Ctimes%20N_%7BVocabulary%7D%7D">

<img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20P%28w_%7Bi%7D%5Cmid%20Non-spam%29%20%3D%20%5Cfrac%7BN_%7Bw_%7Bi%7D%5Cmid%20Non-spam%7D%20&plus;%20%5Calpha%20%7D%7BN_%7BNon-spam%7D%20&plus;%20%5Calpha%5Ctimes%20N_%7BVocabulary%7D%7D">

Where <img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20N_%7BVocabulary%7D">  represents the number of unique words in all the messages. <img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20N_%7Bw_%7Bi%7D%5Cmid%20Non-spam%7D"> is equal to the number of times the word w<sub>i</sub> occurs in spam messages. <img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20N_%7Bw_%7Bi%7D%5Cmid%20Spam%7D"> is equal to the number of times the word w<sub>i</sub> occurs in non-spam messages. <img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20N_%7BSpam%7D"> is equal to the number of words in all the spam messages. <img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%5Clarge%20N_%7BNon-spam%7D"> is equal to the number of words in all the non-spam messages.

If __P(Spam|New SMS) > P(Non-spam|New SMS)__ then algorithm will classify the new message as spam.
This is a [multinomial Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes) version of the algorithm.

We'll apply the algorithm to a [dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) of over 5,000 SMS messages. This is a SMS Spam Collection Data Set from [Maching learning repository](https://archive.ics.uci.edu/ml/index.php). The collection is a text file, where each line has the correct class followed by the raw message.
