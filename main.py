import python_speech_features as psf
from scipy.io import wavfile

NO_SPEAKERS = 10


#Load in data from .wav files in data/
#Extract mfcc (first 13 coefficients) from each audio sample
spk  = [wavfile.read('data/english' + str(i) + '.wav') for i in range(1, NO_SPEAKERS + 1)]
spk_mfcc = [psf.mfcc(spk[i][1], spk[i][0])  for i in range(0, NO_SPEAKERS)]


#Cepstral Mean Subtraction (Feature Normalization step)
from functools import reduce
for i, speaker_mfcc in enumerate(spk_mfcc):
    average = reduce(lambda acc, ele: acc + ele, speaker_mfcc)
    average = list(map(lambda x: x/len(speaker_mfcc), average))
    for j, feature_vector in enumerate(speaker_mfcc):
        for k, feature in enumerate(feature_vector):
            spk_mfcc[i][j][k] -= average[k]


#Create GMM UBM Models for each speaker and prepare data for training
from sklearn.mixture import GaussianMixture

GMM = []
UBM = []

#  Create a GMM and UBM model for each speaker. The GMM is modelled after the speaker and UBM for each speaker
#  is modelled after all the other speakers. Likelihood Ratio test is used to verify speaker
def setGMMUBM(no_components):
    global GMM, UBM
    GMM = []
    UBM = []
    for i in range(NO_SPEAKERS):
        GMM.append(GaussianMixture(n_components= no_components, covariance_type= 'diag'))
        UBM.append(GaussianMixture(n_components= no_components, covariance_type= 'diag'))


# Partition data into training and testing
total_mfcc = []
speaker_label = []
spk_train_size = []
spk_start = []
spk_end = []

for i in range(NO_SPEAKERS):
    spk_train_size.append(int(0.9 * len(spk_mfcc[i])))
    spk_start.append(len(total_mfcc))
    print(i)
    for mfcc in spk_mfcc[i][0:spk_train_size[i], :]:
        total_mfcc.append(mfcc)
        speaker_label.append(i)
    spk_end.append(len(total_mfcc))


# Fit the GMM UBM models with training data
def fit_model():
    for i in range(NO_SPEAKERS):
        print("Fit start for {}".format(i))
        GMM[i].fit(spk_mfcc[i])
        UBM[i].fit(total_mfcc[:spk_start[i]] + total_mfcc[spk_end[i]:])
        print("Fit end for {}".format(i))


# Predict the output for each model for each speaker and produce confusion matrix
def execute():
    fit_model()
    avg_accuracy = 0

    confusion = [[ 0 for y in range(NO_SPEAKERS) ] for x in range(NO_SPEAKERS)]

    for i in range(NO_SPEAKERS):
        for j in range(NO_SPEAKERS):
            x = GMM[j].score_samples(spk_mfcc[i][spk_train_size[i] + 2 : ]) - UBM[j].score_samples(spk_mfcc[i][spk_train_size[i] + 2 : ])
            for score in x :
                if score > 0:
                    confusion[i][j] += 1

    confusion_diag = [confusion[i][i] for i in range(NO_SPEAKERS)]

    diag_sum = 0
    for item in confusion_diag:
        diag_sum += item

    remain_sum = 0
    for i in range(NO_SPEAKERS):
        for j in range(NO_SPEAKERS):
            if i != j:
                remain_sum += confusion[i][j]

    spk_accuracy = 0
    for i in range(NO_SPEAKERS):
        best_guess, _ = max(enumerate(confusion[i]), key=lambda p: p[1])
        print("For speaker {}, best guess is {}".format(i, best_guess))
        if i == best_guess:
            spk_accuracy += 1
    spk_accuracy /= NO_SPEAKERS

    avg_accuracy = diag_sum/(remain_sum+diag_sum)
    return confusion, avg_accuracy, spk_accuracy


#TBD : Ten fold validation
def ten_fold():
    #fold_size = 0.1 * self.n
    fold_offset = 0.0
    accuracy_per_fold = 0
    average_accuracy = 0

    for i in range(0, 10):
        print("Fold start is {}  and fold end is {} ".format( fold_offset, fold_offset + fold_size))
        #accuracy = self.execute(int(fold_offset), int(fold_offset + fold_size))
        #print("Accuracy is of test {} is : {} ".format(i, accuracy))
        #average_accuracy += accuracy
        #fold_offset += fold_size

    average_accuracy /= 10.0
    print("Average accuracy  " + str(100 * average_accuracy))
    return average_accuracy

# Gaussian Mixture Model is made of a number of Gaussian distribution components. To model data, a suitable number of
# gaussian components have to be selected. There is no method for finding this. It is done by trial and error. This runs
# the program for different values of component and records accuracy for each one
def find_best_params():
    best_no_components = 1
    maxacc = 0
    for i in range(1, 256):
        setGMMUBM(i)
        acc, _ = execute()
        print("Accuracy for n = {} is {}".format(i, acc))
        if acc > maxacc:
            maxacc = acc
            best_no_components = i
    return best_no_components

# Final result is a confusion matrix which represents the accuracy of the fit of the model
import numpy as np
if __name__ == '__main__':
    setGMMUBM(244)
    confusion, mfcc_accuracy, spk_accuracy = execute()
    print("Confusion Matrix")
    print(np.matrix(confusion))
    print("Accuracy in predicting speakers : {}".format(spk_accuracy))
    print("Accuracy in testing for MFCC : {}".format(mfcc_accuracy))