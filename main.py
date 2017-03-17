import python_speech_features as psf
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib
from scipy.io import wavfile
from functools import reduce

MODEL_SPEAKERS = 50
TOTAL_SPEAKERS = 50

class SpeakerRecognition:

    #  Create a GMM and UBM model for each speaker. The GMM is modelled after the speaker and UBM for each speaker
    #  is modelled after all the other speakers. Likelihood Ratio test is used to verify speaker
    def setGMMUBM(self, no_components):
        for i in range(MODEL_SPEAKERS):
            self.GMM.append(GaussianMixture(n_components= no_components, covariance_type= 'diag'))
            self.UBM.append(GaussianMixture(n_components= no_components, covariance_type= 'diag'))

    # Load in data from .wav files in data/
    # Extract mfcc (first 13 coefficients) from each audio sample
    def load_data(self):
        self.spk  = [wavfile.read('data/english' + str(i) + '.wav') for i in range(1, TOTAL_SPEAKERS + 1)]
        self.spk_mfcc = [psf.mfcc(self.spk[i][1], self.spk[i][0])  for i in range(0, TOTAL_SPEAKERS)]

        for i in range(TOTAL_SPEAKERS):
            self.spk_train_size.append(int(0.9 * len(self.spk_mfcc[i])))
            self.spk_start.append(len(self.total_mfcc))
            print(i)
            for mfcc in self.spk_mfcc[i][0:self.spk_train_size[i], :]:
                self.total_mfcc.append(mfcc)
                self.speaker_label.append(i)
            self.spk_end.append(len(self.total_mfcc))



    # Gaussian Mixture Model is made of a number of Gaussian distribution components.
    #  To model data, a suitable number o gaussian components have to be selected.
    # There is no method for finding this. It is done by trial and error. This runs
    # the program for different values of component and records accuracy for each one
    def find_best_params(self):
        best_no_components = 1
        maxacc = 0
        for i in range(1, 256):
            self.setGMMUBM(i)
            acc, _ = self.execute()
            print("Accuracy for n = {} is {}".format(i, acc))
            if acc > maxacc:
                maxacc = acc
                best_no_components = i
        return best_no_components

    # Fit the GMM UBM models with training data
    def fit_model(self):
        for i in range(MODEL_SPEAKERS):
            print("Fit start for {}".format(i))
            self.GMM[i].fit(self.spk_mfcc[i])
            self.UBM[i].fit(self.total_mfcc[:self.spk_start[i]] + self.total_mfcc[self.spk_end[i]:])
            print("Fit end for {}".format(i))
            joblib.dump(self.UBM[i], 'dumps/ubm' + str(i) + '.pkl')
            joblib.dump(self.GMM[i], 'dumps/gmm' + str(i) + '.pkl')

    def model(self, no_components = 244):
        self.setGMMUBM(no_components)
        self.fit_model()

    # Predict the output for each model for each speaker and produce confusion matrix
    def load_model(self):
        for i in range(0, MODEL_SPEAKERS):
            self.GMM.append(joblib.load('dumps/gmm' + str(i) + '.pkl'))
            self.UBM.append(joblib.load('dumps/ubm' + str(i) + '.pkl'))

    def predict(self):
        avg_accuracy = 0

        confusion = [[ 0 for y in range(MODEL_SPEAKERS) ] for x in range(TOTAL_SPEAKERS)]

        for i in range(TOTAL_SPEAKERS):
            for j in range(MODEL_SPEAKERS):
                x = self.GMM[j].score_samples(self.spk_mfcc[i][self.spk_train_size[i] + 2 : ]) - self.UBM[j].score_samples(self.spk_mfcc[i][self.spk_train_size[i] + 2 : ])
                for score in x :
                    if score > 0:
                        confusion[i][j] += 1

        confusion_diag = [confusion[i][i] for i in range(MODEL_SPEAKERS)]

        diag_sum = 0
        for item in confusion_diag:
            diag_sum += item

        remain_sum = 0
        for i in range(MODEL_SPEAKERS):
            for j in range(MODEL_SPEAKERS):
                if i != j:
                    remain_sum += confusion[i][j]

        spk_accuracy = 0
        for i in range(MODEL_SPEAKERS):
            best_guess, _ = max(enumerate(confusion[i]), key=lambda p: p[1])
            print("For speaker {}, best guess is {}".format(i, best_guess))
            if i == best_guess:
                spk_accuracy += 1
        spk_accuracy /= MODEL_SPEAKERS

        avg_accuracy = diag_sum/(remain_sum+diag_sum)
        return confusion, avg_accuracy, spk_accuracy

    def __init__(self):
        self.test_spk = []
        self.test_mfcc = []

        # Speaker data and corresponding mfcc
        self.spk = []
        self.spk_mfcc = []

        # Holds all the training mfccs of all speakers and
        # speaker_label is the speaker label for the corresponding mfcc

        self.total_mfcc = []
        self.speaker_label = []
        self.spk_train_size = []  # Index upto which is training data for that speaker.

        # Since the length of all the audio files are different, spk_start and spk_end hold

        self.spk_start = []
        self.spk_end = []

        self.GMM = []
        self.UBM = []
        self.load_data()
        self.cepstral_mean_subtraction()

    # Cepstral Mean Subtraction (Feature Normalization step)
    def cepstral_mean_subtraction(self):
        for i, speaker_mfcc in enumerate(self.spk_mfcc):
            average = reduce(lambda acc, ele: acc + ele, speaker_mfcc)
            average = list(map(lambda x: x/len(speaker_mfcc), average))
            for j, feature_vector in enumerate(speaker_mfcc):
                for k, feature in enumerate(feature_vector):
                    self.spk_mfcc[i][j][k] -= average[k]



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



# Final result is a confusion matrix which represents the accuracy of the fit of the model
import numpy as np
if __name__ == '__main__':

    SR = SpeakerRecognition()
    SR.load_model()
    confusion, mfcc_accuracy, spk_accuracy = SR.predict()

    print("Confusion Matrix")
    print(np.matrix(confusion))
    print("Accuracy in predicting speakers : {}".format(spk_accuracy))
    print("Accuracy in testing for MFCC : {}".format(mfcc_accuracy))