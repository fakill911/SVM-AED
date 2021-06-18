import os,pyaudio,time,numpy,csv,librosa,pickle,sys #,pylab
from playsound import playsound
from collections import Counter
import pyqtgraph as pg
from python_speech_features import mfcc, delta ,sigproc#,get_filterbanks,fbank
from sklearn.pipeline import Pipeline
from multiprocessing import Process
import noisereduce as nr
from scipy.stats import skew
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from joblib import dump, load
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split,StratifiedKFold,KFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
# import scipy.stats
# from thundersvm import SVC
import scipy.signal as signal
import wave
#import panda
from sklearn.model_selection import GridSearchCV
#import sounddevice as sd
from sklearn.model_selection import cross_val_score
os.chdir('C:\\Users\\frenn\\Desktop\\FPA\\PFE\\ESC\\audio')
import librosa.display

def attach_extension(row):
    ext = list(str(row))
    ext.append(".wav")
    ext = ''.join(ext)
    return ext

def feature_extract(f,detection=True):
    SAMPLE_RATE = 16000  # Sample rate
    if detection==True:
        data=f
    else:
        spl = librosa.effects.split(f, top_db=22)
        data = []
        for uns in spl:
            data = numpy.append(data, f[uns[0]:uns[1]])
    mfcc_feature = mfcc(data, samplerate=SAMPLE_RATE, numcep=18, preemph=0.97,winfunc=numpy.hamming,appendEnergy=False,ceplifter=15)
    stat_mfcc_feat = numpy.hstack(
        (numpy.mean(mfcc_feature, axis=0), numpy.std(mfcc_feature, axis=0),
         skew(mfcc_feature, axis=0), numpy.max(mfcc_feature, axis=0), numpy.min(mfcc_feature, axis=0),
         numpy.median(mfcc_feature, axis=0)))

    zcr = librosa.feature.zero_crossing_rate(data,frame_length=int(SAMPLE_RATE*0.025), hop_length=int(SAMPLE_RATE*0.01))[0]
    zcr = numpy.hstack((numpy.mean(zcr, axis=0), numpy.std(zcr, axis=0),
                        skew(zcr, axis=0), numpy.max(zcr, axis=0), numpy.min(zcr, axis=0), numpy.median(zcr, axis=0)))
    delta_one = delta(mfcc_feature, 2)
    delta_two = delta(delta_one, 2)
    delta_one = numpy.hstack(
        (numpy.mean(delta_one, axis=0), numpy.std(delta_one, axis=0),
         skew(delta_one, axis=0), numpy.max(delta_one, axis=0), numpy.min(delta_one, axis=0),
         numpy.median(delta_one, axis=0)))
    delta_two = numpy.hstack(
        (numpy.mean(delta_two, axis=0), numpy.std(delta_two, axis=0),
         skew(delta_two, axis=0), numpy.max(delta_two, axis=0), numpy.min(delta_two, axis=0),
         numpy.median(delta_two, axis=0)))
    features = numpy.hstack((stat_mfcc_feat, delta_one, delta_two, zcr))
    return features

def feature_scale(feature,feat,detection):
    if detection==True:
        sel = VarianceThreshold(threshold=0.16)
        sel=sel.fit(feat)
        selected_all=sel.fit_transform(feat)
        feat_selected=sel.transform(numpy.reshape(feature, (1, -1)))
        scaler = MinMaxScaler()
        scaler = scaler.fit(selected_all)
        feat_2 = scaler.transform(feat_selected)
        return feat_2
    else:
        sel = VarianceThreshold(threshold=0.16)
        sel=sel.fit(feat)
        selected_all=sel.fit_transform(feat)
        feat_selected=sel.transform(numpy.reshape(feature, (1, -1)))
        scaler = MinMaxScaler()
        scaler = scaler.fit(selected_all)
        feat = scaler.transform(feat_selected)
        return feat

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    cm=numpy.round(cm,decimals=2)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=numpy.arange(cm.shape[1]),
           yticks=numpy.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",fontsize=6,
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",fontsize=6)
    return ax

def feature_compute(j):
    feat=[]
    label_num=[]
    i=2000
    with open('background.csv',newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        start_time = time.time()
        SAMPLE_RATE = 16000
        #noisy_part, _ = librosa.core.load("468.wav", sr=SAMPLE_RATE, res_type='kaiser_fast')
        for row in spamreader:
            #record_background(i, row[0])
            # ext = attach_extension(row[0])
            silent_data, _ = librosa.core.load(str(row[0]), sr=SAMPLE_RATE,res_type='kaiser_fast')
            spl=librosa.effects.split(silent_data,top_db=22) #22
            data=[]
            #write(str(attach_extension(i)), rate=16000, data=silent_data)
            for uns in spl:
                data = numpy.append(data,silent_data[uns[0]:uns[1]])
            #naem = str(attach_extension(i))
            #write(naem, SAMPLE_RATE, data)
            '''if row[2]=='0':
                data = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=False)
                print("this")'''
            print(row[2])
            '''data=preemphasis(data,coeff=0.97)
            framed=sigproc.framesig(data,551.25,220.5,winfunc=numpy.hamming)
            s=powspec(frames=framed, NFFT=512)
            plt.figure(10)
            Pxx, freqs =plt.psd(framed[10],Fs=SAMPLE_RATE,NFFT=512)
            plt.figure(11)
            plt.plot(freqs,s[10],zorder=100,linewidth=3)
            mlbnk=get_filterbanks(nfilt=12,samplerate=SAMPLE_RATE,nfft=512)
            i=0

            while i<12:
                plt.plot(freqs,mlbnk[i])
                i+=1
            prod=numpy.dot(s,mlbnk.T)
            print(prod)
            fbb, _=fbank(data,samplerate=SAMPLE_RATE,nfft=512,nfilt=12)
            print("_________________________________________________________________________")
            print(fbb)
            plt.ylabel("Magnitude")
            plt.xlabel("Frequency")
            plt.title('Mel filterbanks')
            plt.show()'''
            # ext=attach_extension(row[5])
            # scipy.io.wavfile.write(ext,rate=44100,data=data)
            mfcc_feature = mfcc(data, samplerate=SAMPLE_RATE, numcep=18, preemph=0.97,winfunc=numpy.hamming,appendEnergy=False,ceplifter=15)
            '''print(mfcc_feature[0])
            plt.figure(figsize=(10, 4))
            mf=numpy.transpose(mfcc_feature)
            librosa.display.specshow(mf, sr=SAMPLE_RATE,x_axis='frames')
            plt.colorbar()
            plt.title('MFCC')
            plt.tight_layout()
            plt.ylabel("Coefficients")
            plt.show()'''
            stat_mfcc_feat = numpy.hstack((numpy.mean(mfcc_feature, axis=0), numpy.std(mfcc_feature, axis=0),
                                           skew(mfcc_feature, axis=0), numpy.max(mfcc_feature, axis=0),numpy.min(mfcc_feature, axis=0),
                                           numpy.median(mfcc_feature, axis=0)))
            # stat_mfcc_feat size is (7,13), 13 mfcc features, 7 is the number of stastistical characterstics
            # we used to reduce all the frames, in order to compute the model more efficiently)
            zcr = librosa.feature.zero_crossing_rate(data,frame_length=int(SAMPLE_RATE*0.025), hop_length=int(SAMPLE_RATE*0.01))[0]
            '''z=numpy.linspace(0,len(data)/ SAMPLE_RATE, num=len(zcr))
            Time = numpy.linspace(0, len(data) / SAMPLE_RATE, num=len(data))
            plt.figure(2)
            plt.title('Audio signal')
            plt.plot(Time, data,'r',zorder=0)
            plt.scatter(z,zcr,c='g',zorder=1)
            plt.ylabel("Amplitude")
            plt.xlabel("Time")
            plt.show()'''
            zcr = numpy.hstack((numpy.mean(zcr, axis=0), numpy.std(zcr, axis=0),
                                skew(zcr, axis=0),numpy.max(zcr, axis=0), numpy.min(zcr, axis=0), numpy.median(zcr, axis=0)))
            delta_one = delta(mfcc_feature, 2)
            delta_two = delta(delta_one, 2)
            delta_one = numpy.hstack((numpy.mean(delta_one, axis=0), numpy.std(delta_one, axis=0),
                                      skew(delta_one, axis=0), numpy.max(delta_one, axis=0),numpy.min(delta_one, axis=0),
                                      numpy.median(delta_one, axis=0)))
            delta_two = numpy.hstack((numpy.mean(delta_two, axis=0), numpy.std(delta_two, axis=0),
                                      skew(delta_two, axis=0), numpy.max(delta_two, axis=0),numpy.min(delta_two, axis=0),
                                      numpy.median(delta_two, axis=0)))
            features = numpy.hstack((stat_mfcc_feat, delta_one, delta_two, zcr))
            # numpy.savetxt(str(ext), mfcc_feat, delimiter=',')
            if numpy.size(feat) == 0:
                feat = features
                label_num = numpy.append(label_num, row[2])
                # label_num = numpy.append(label_num, row[3])
            else:
                feat = numpy.vstack((feat, features))
                # label_num = numpy.append(label_num, row[3])
                label_num = numpy.append(label_num, row[2])
            i+=1
        print("--- %s seconds ---" % (time.time() - start_time))
        feat.dump("dfeat")
        label_num.dump("dlabel_num")
    return feat,label_num

def model_learn():
    jay=[]
    j=38
    average_cv=[]
    while j<39:
        #feat, label_num = feature_compute(j)
        feat = numpy.load("dfeat", allow_pickle=True)
        label_num = numpy.load("dlabel_num", allow_pickle=True)
        sel = VarianceThreshold(threshold=0.16)
        print(numpy.size(feat,1))
        feat=sel.fit_transform(feat)
        print(numpy.size(feat,1))
        scaler = MinMaxScaler()
        #print(numpy.var(feat, axis=0))
        #joblib.dump(scaler.fit(feat), "Scaler")
        X_train, X_test, y_train, y_test = train_test_split(feat, label_num, test_size=0.33,random_state=45,stratify=label_num)
        d=scaler.fit(feat)
        feat=scaler.transform(feat)
        #print(y_test)
        #print(Counter(y_test))
        fold=StratifiedKFold(n_splits=10,shuffle=True,random_state=45)
        '''C_grid = [0.001, 0.01, 0.1, 1, 10]
        gamma_grid = [0.001, 0.01, 0.1, 1, 10]
        param_grid = {'C': C_grid, 'gamma': gamma_grid}
        grid = GridSearchCV(SVC(kernel='rbf', decision_function_shape='ovo'), param_grid, cv=fold, scoring="accuracy")
        grid.fit(feat, label_num)
        print(grid.cv_results_)
        print(grid.best_score_)
        print(grid.best_params_)
        print(grid.best_estimator_)'''
        clf= SVC(C=1,gamma=0.1,verbose=False,kernel='rbf',probability=True,decision_function_shape='ovr')
        cv=cross_val_score(clf, feat, label_num, cv=fold)
        average_cv=numpy.append(average_cv, numpy.sum(cv)/10)
        jay=numpy.append(jay, j)
        print(jay)
        print(average_cv)
        print(numpy.max(average_cv))
        clf.fit(feat, label_num) #--------------------------------------------------------
        # y_pred = clf.predict(numpy.reshape(X_test[0], (1, -1)))
        dump(clf, 'dmymodel.joblib')
        '''y=clf.predict(X_test)
        label = ['dog', 'cow', 'frog', 'crow', 'crackling_fire', 'water_drops', 'pouring_water', 'thunderstorm','crying_baby',
         'sneezing','clapping', 'coughing', 'laughing', 'drinking_sipping', 'snoring', 'door_wood_knock','keyboard_typing',
          'door_wood_creaks','clock_alarm', 'glass_breaking', 'helicopter', 'siren', 'car_horn', 'train', 'church_bells']
        plot_confusion_matrix(y_test, y, classes=label, normalize=True,title='Normalized confusion matrix')
        plt.show()'''
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        j+=1

def model_application():
    SAMPLE_RATE=16000
    label=['dog','cow','frog','crow','crackling_fire','water_drops','pouring_water','thunderstorm','crying_baby',
           'sneezing', 'clapping','coughing','laughing','snoring','drinking_sipping','door_wood_knock','keyboard_typing',
           'door_wood_creaks','clock_alarm','glass_breaking','helicopter','siren','car_horn','train','church_bells']
    clf = load('mymodel.joblib')
    clf_2 = load('dmymodel.joblib')
    data_feat_classify = numpy.load("feat", allow_pickle=True)
    data_feat_detect = numpy.load("dfeat", allow_pickle=True)
    data, _ = librosa.core.load("youtube.wav", sr=SAMPLE_RATE, res_type='kaiser_fast',mono=True)
    #noisy_part, _ = librosa.core.load("background.wav", sr=SAMPLE_RATE, res_type='kaiser_fast')
    #data = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=False)
    # write("check.wav", SAMPLE_RATE, data)
    overlap=0.1
    win_len=1
    framed = sigproc.framesig(data, SAMPLE_RATE*win_len, SAMPLE_RATE*overlap, winfunc=numpy.hamming)
    n_start =0
    event_time=[]
    event_time_end=[]
    for f in framed:
        feature_2 = feature_extract(f,detection=True)
        feat_2 = feature_scale(feature_2,data_feat_detect,detection=True)
        event_detected = clf_2.predict(feat_2)
        if event_detected=='1':
            feature = feature_extract(f,detection=False) # fix silence removal in feat trained silence removal is only for events add liftering for detection?
            feat = feature_scale(feature,data_feat_classify, detection=False)
            y = clf.predict(feat)
            print("event detected between: ", round(n_start * overlap,2), " and ", round(n_start * overlap + win_len,2), "I'm most likely hearing: ",label[int(y)])
            event_time = numpy.append(event_time,n_start * overlap)
            event_time_end = numpy.append(event_time_end, n_start * overlap +win_len)
            pr = clf.predict_proba(feat)[0]
            ind = numpy.argsort(pr)[-3:]
            #print("I am most likely hearing:")
            #print(label[ind[2]], round(pr[ind[2]] * 100, 2), "% \n", label[ind[1]], round(pr[ind[1]] * 100, 2),"% \n",label[ind[0]], round(pr[ind[0]] * 100, 2), "%")

        else:
            print("no event between: ", round(n_start * overlap,2), " and ", round(n_start * overlap + win_len,2))
        n_start += 1
    print(event_time)
    Time = numpy.linspace(0, len(data) / SAMPLE_RATE, num=len(data))
    plt.plot(Time,data)
    plt.ylim((-1, 1))
    plt.xlim((0,len(data)/SAMPLE_RATE))
    for et, ee in zip(event_time, event_time_end):
        plt.axvspan(et, ee, color='red', alpha=0.5)
    #t=str(label[ind[2]]) + str(round(pr[ind[2]] * 100, 2)) + " % \n"+ str(label[ind[1]])+ str(round(pr[ind[1]] * 100, 2)) + " % \n" + str(label[ind[0]]) + str(round(pr[ind[0]] * 100, 2)) + " %"
    #plt.text(2, 0.25, t,rotation='vertical',color='black', fontsize=12)
    plt.show()

    '''p = numpy.array(clf.decision_function(feat))  # decision is a voting function
    prob = numpy.exp(p) / numpy.sum(numpy.exp(p), axis=1)  # softmax after the voting
    classes = clf.predict(feat)
    _ = [print('Sample={}, Prediction={},\n Votes={} \nP={}, '.format(idx, c, v, s)) for idx, (v, s, c) in
         enumerate(zip(p, prob, classes))]'''

    '''pr=clf.predict_proba(feat)[0]
    ind = numpy.argsort(pr)[-3:]
    print("I am most likely hearing:")
    print(label[ind[2]],round(pr[ind[2]]*100,2),"% \n",label[ind[1]],round(pr[ind[1]]*100,2),"% \n",label[ind[0]],
          round(pr[ind[0]]*100,2),"%")'''
    return
def record_background(j,row):

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 6
    WAVE_OUTPUT_FILENAME = attach_extension(j)
    print(WAVE_OUTPUT_FILENAME)
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("* recording")
    frames = []
    playsound(str(row),block = False)
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    time.sleep(1)


CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()
fulldata = numpy.array([])
audio_data = numpy.array([])
Times=numpy.array([])
CHUNK=1024
def nein():
    global fulldata,Times,audio_data
    stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                input=True,
                stream_callback=callback,
                frames_per_buffer=CHUNK)
    stream.start_stream()
    print("initializing")
    while stream.is_active():
        time.sleep(10)
        stream.stop_stream()
    stream.close()
    write("ok.wav", rate=16000, data=fulldata)
    p.terminate()


def callback(in_data, frame_count, time_info, flag):
    global fulldata,Times,audio_data
    audio_data = numpy.frombuffer(in_data, dtype=numpy.float32)
    fulldata = numpy.append(fulldata, audio_data)
    Times = numpy.linspace(0, len(fulldata) / RATE, num=len(fulldata))
    return (audio_data, pyaudio.paContinue)


'''def rt_plot():
    global Times, fulldata,audio_data
    while True:
        ani=FuncAnimation(plt.gcf(),animate,interval=500)
        plt.plot(Times,fulldata)
        print(audio_data)'''


#feature_compute(5)
#model_learn()
#record_background()
model_application()
#nein()

