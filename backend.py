# Importieren der Programmbibliotheken
from flask import Flask, redirect, url_for, request, render_template, send_from_directory
from werkzeug import secure_filename
import librosa
import librosa.display
import soundfile as sf
import IPython.display
import numpy as np
import pandas as pd
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import vgg16
from keras.models import Model
import keras
import matplotlib.pyplot as plt
import os, shutil
import mysql.connector
import string
from pydub import AudioSegment

# Erstellung einer Anwendungsvariable app
app = Flask(__name__)

# Verbindung zur Datenbank
mydb = mysql.connector.connect(
   host =  "localhost",
   user = "root",
   passwd = "root",
   database = "sounddb"
)

mycursor = mydb.cursor()
mycursor.execute("use soundschema")

# Erstellung bzw. Wiedeverwendung der Klasse AudioIdentifier
class AudioIdentifier: 
    # Methode zur Initialisierung des Modells   
    def __init__(self, prediction_model_path=None):
        self.vgg_model = self.load_vgg_model()
        self.prediction_model = self.load_prediction_model(prediction_model_path)        
        self.class_map = {'0' : 'Saxophone', '1' : 'Violin_or_fiddle', '2' : 'Hi-hat', '3' : 'Acoustic_guitar', '4' : 'Double_bass', 
                 '5' : 'Cello', '6' : 'Bass_drum', '7' : 'Flute', '8' : 'Clarinet', '9' : 'Snare_drum'}
        self.predicted_class = None
        self.predicted_label = None
        
    # laden vom vgg-16 Model
    def load_vgg_model(self):
        vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                     input_shape=(64, 64, 3))
        output = vgg.layers[-1].output
        output = keras.layers.Flatten()(output)
        model = Model(vgg.input, output)
        model.trainable = False
        return model
    
    # laden vom Vorhersage Model
    def load_prediction_model(self, model_path):
        pred_model = keras.models.load_model(model_path)
        return pred_model
    
    # Erhalten der Audio Informationen in der benötigten Sample-Rate    
    def get_sound_data(self, path, sr=22050):
        data, fsr = sf.read(path)
        data_22k = librosa.resample(data.T, fsr, sr)
        if len(data_22k.shape) > 1:
            data_22k = np.average(data_22k, axis=0)
            
        return data_22k, sr
    
    # Zeitfenster für die Unter-Sample (Sub-Sample)
    def windows(self, data, window_size):
        start = 0
        while start < len(data):
            yield int(start), int(start + window_size)
            start += (window_size / 2) 
            
    # Extrahierung der 3D-Feature-Maps aus den Unter-Sample        
    def extract_base_features(self, sound_data, bands=64, frames=64):
    
        window_size = 512 * (frames - 1)  
        log_specgrams_full = []
        log_specgrams_hp = []
        
        start, end = list(self.windows(sound_data, window_size))[0]
        
        if(len(sound_data[start:end]) == window_size):
            signal = sound_data[start:end]

            melspec_full = librosa.feature.melspectrogram(signal, n_mels = bands)
            #logspec_full = librosa.logamplitude(melspec_full)#nicht aktuell
            logspec_full = librosa.amplitude_to_db(melspec_full)#aktuell
            logspec_full = logspec_full.T.flatten()[:, np.newaxis].T

            y_harmonic, y_percussive = librosa.effects.hpss(signal)
            melspec_harmonic = librosa.feature.melspectrogram(y_harmonic, n_mels = bands)
            melspec_percussive = librosa.feature.melspectrogram(y_percussive, n_mels = bands)
            #logspec_harmonic = librosa.logamplitude(melspec_harmonic)#nicht aktuell
            logspec_harmonic = librosa.amplitude_to_db(melspec_harmonic)#aktuell
            #logspec_percussive = librosa.logamplitude(melspec_percussive)#nicht aktuell
            logspec_percussive = librosa.amplitude_to_db(melspec_percussive)#aktuell
            logspec_harmonic = logspec_harmonic.T.flatten()[:, np.newaxis].T
            logspec_percussive = logspec_percussive.T.flatten()[:, np.newaxis].T
            logspec_hp = np.average([logspec_harmonic, logspec_percussive], axis=0)

            log_specgrams_full.append(logspec_full)
            log_specgrams_hp.append(logspec_hp)

        log_specgrams_full = np.asarray(log_specgrams_full).reshape(len(log_specgrams_full), bands ,frames, 1)
        log_specgrams_hp = np.asarray(log_specgrams_hp).reshape(len(log_specgrams_hp), bands ,frames, 1)
        features = np.concatenate((log_specgrams_full, 
                                   log_specgrams_hp, 
                                   np.zeros(np.shape(log_specgrams_full))), 
                                  axis=3)

        for i in range(len(features)):
            features[i, :, :, 2] = librosa.feature.delta(features[i, :, :, 0])

        return np.array(features)
    
    # Extrahierung der Transfer Learning Feature
    def extract_transfer_learn_features(self, base_feature_data):
        
        base_feature_data = np.expand_dims(base_feature_data, axis=0)
        base_feature_data = preprocess_input(base_feature_data)
        model = self.vgg_model
        tl_features = model.predict(base_feature_data)
        tl_features = np.reshape(tl_features, tl_features.shape[1])
        return tl_features
    
    # Erzeugung der Methode Feature-Engineering 
    def feature_engineering(self, audio_data):
        base_features = self.extract_base_features(sound_data=audio_data)
        final_feature_map = self.extract_transfer_learn_features(base_features[0])
        return final_feature_map
    
    # Methode für die Klassen-Vorhersage
    def prediction(self, feature_map):
        model = self.prediction_model
        feature_map = feature_map.reshape(1, -1)
        pred_class = model.predict_classes(feature_map, verbose=0)
        return pred_class[0]
        
    # Erhalten der Klassifizierung der Audiodatei
    def prediction_pipeline(self, audio_file_path, return_class_label=True):
        
        audio_data, sr = self.get_sound_data(audio_file_path)
        feature_map = self.feature_engineering(audio_data)
        prediction_class = self.prediction(feature_map)
        self.predicted_class = prediction_class
        self.predicted_label = self.class_map[str(self.predicted_class)]  
        if return_class_label:
            return self.predicted_label
        else:
            return self.predicted_class


#ai = AudioIdentifier(prediction_model_path='sound_classification_model.h5')

class_map = {'Bass_drum' : 1, 'Snare_drum': 2, 'Hi-hat': 3}
#class_map2 = {1 : 'Bass_drum', 2 : 'Snare_drum', 3 : 'Hi-hat'}

# Ermittlung der Länge der Audiodatei in Sek.
def soundLength(pfadDat):
    #pfadDat = "./wavT/MaxV - HH Cl.wav"
    #pfadDat = "MaxV_-_HH_Cl.wav"
    print(pfadDat)
    f = sf.SoundFile(pfadDat)     
    soundL = (len(f)/f.samplerate)
    soundL = round(soundL, 2)
    return soundL

# Verlängerung der Audiodatei um eine Stille-Zeit
def soundMitStille(inSound, outSound):#file mit Ordner als String übergeben 
    #inSound = "./wavT/MaxV - HH Cl.wav"#übergabeparamete in inSound speichern
    #outSound = "./zielHHOrdner/out.wav"
    print(soundLength(inSound))
    #if soundlänge < 1.5, dann Anpassung
    #else keine Anpassung
    
    # create 1 sec of silence audio segment
    #one_sec_segment = AudioSegment.silent(duration=1000)  #duration in milliseconds    
    one_sec_segment = AudioSegment.silent(duration=250)

    while(soundLength(inSound)<1.5):        
        #read wav file to an audio segment
        song = AudioSegment.from_wav(inSound)

        #Add above two audio segments    
        final_song = song+one_sec_segment

        #Either save modified audio
        final_song.export(outSound, format="wav")
    
    print(soundLength(outSound))

#Abruf gleichartiger sounds
@app.route('/abruf/<pfdDat>/<zielOrdner>')
def abruf(pfdDat, zielOrdner):

    # Entwicklungskommentare
        #pfadDat = "../wavT/MaxV - HH Cl.wav"#Auswahlmenü machen oder auswhähl/eintipen Dateipfad
        #pfadDat = "./wavT/MaxV - Snare.wav"
        # Vor Pred(Predictio) immer Sound länge checken und evtl. verlängern
        #mit Rückgabe von soundMitStille weiterarbeiten


    pfadDat = os.path.relpath(pfdDat)
    # Entwicklungskommentare
        #return pfadDat
        #pfadDat = "MaxV_-_HH_Cl.wav"
    print(os.path.isfile(pfadDat))

    # Prüfung auf Dateilänge
    if(soundLength(pfadDat)<1.5):
        #sound verlängern
        soundMitStille(pfadDat, pfadDat)
    
    else:
        print("ist nicht < 1.5") 

    # Erstellung von einem Objekt der Klasse AudioIdentifier mit dem vorher Trainierten Modell 
    ai = AudioIdentifier(prediction_model_path='sound_classification_model.h5')    
    pred = ai.prediction_pipeline(pfadDat, return_class_label=True)
    
    # Entwicklungskommentare
        #Quell und Ziel -Pfade auf Aktuallität prüfen ganz wichtig in der DB !!!!!
    
    # Rückmeldung in der Konsole, wenn das Instrument nicht Erkannt wurde
    # Rückmeldung der möglichen Instrumentes (Klasse)        
    if(class_map.get(pred,"nix")=="nix"):        
        return pfadDat+" ist kein Drum-Sample, Prediction: "+str(pred)
    else:
        # Abruf der gleich-klingenden Audiodateien
        sqlFormula = "SELECT dateiPfad FROM sounddatei WHERE lid ="+str(class_map.get(pred))        
        mycursor.execute(sqlFormula)
        myresult = mycursor.fetchall()
        dest = os.path.realpath(zielOrdner)        
        os.chdir('D:\BA2_AudioProj\Drum-samples-classification\Audio-Classification-master')
        #eingeben auf Webseite
        print(os.getcwd())#ganz wichtig, Prüfung ob eingegebener Pfad verwendbar ist
        for row in myresult:            
            sourceP = str(row)
            st = sourceP.find("'")
            end = sourceP.find("'",st+1,-1)
            sourceP = sourceP[st+1:end]             
            shutil.copy2(sourceP, dest)
        mydb.close()
    return 'Copy and Classify Done siehe Zielordner'# Erfolgreiche Klassifikation und Erweiterung der Audio-Datenbank
    
# Eingabe der gesuchten Datei und Zielordner der gesuchten Dateien 
@app.route('/abrufDialog', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':        
        dateiPfad = request.form['dateiPfad']
        #anzSounds = request.form['anzSounds']#Ergänzen darauf abfragen
        zielPfad = request.form['zielPfad']       
        return redirect(url_for('abruf',pfdDat = dateiPfad, zielOrdner = zielPfad))

# Ausgabe vom Ergebnis der Klassifikation der neuen Audio-Dateien als Tabelle
@app.route('/tabelleAusgeben')
def gebeTabAus():
    #Select Befehl ergänzen mit Datum und Uhrzeit bzw. Ausblick
    #Im Ausblick die letzten gespeicherten ausgeben
    sqlFormula = "SELECT dateiPfad, lid FROM sounddatei"
    mycursor.execute(sqlFormula)
    rows = mycursor.fetchall()    
    mydb.close()
    return render_template('result.html', rows = rows) 
    
    #content_type='application/json')später    
    #return 'Hallo'

# Sortierung der Audio-Dateien eines Ordners und Erweiterung der Datenbank
@app.route('/sortierer/<soundOrdner>')
def sortiereSound(soundOrdner):
    pfad = soundOrdner # Speicherung der Pfad vom Ordner
    print(os.path.isdir(pfad)) # Prüfung auf Korrektheit des Pfades
    #pfad = './wavT/'#variabel später  
    ai = AudioIdentifier(prediction_model_path='sound_classification_model.h5')  # Erstellung einer Instand der Klasse AudioIdentifier
    sqlformula = "INSERT INTO sounddatei (dateipfad, lid) values (%s, %s)" # SQL Insert Befehl für die Klassifizierten Audio-Dateien
    
    # Alle Dateien des Ordners durchgehen
    for dirName, subdirList, fileList in os.walk(pfad):
            for file in fileList:                
                # Vor Pred(Predictio) immer Sound länge checken und evtl. verlängern
                #mit Rückgabe von soundMitStille weiterarbeiten
                if(soundLength(dirName+os.sep+file)<1.5): # Prüfung der Dateilänge
                    #sound verlängern
                    soundMitStille(dirName+os.sep+file, dirName+os.sep+file)
                #else:
                    #print("ist nicht < 1.5") 
                    #soundlänge OK                              
                pred = ai.prediction_pipeline(dirName+os.sep+file, return_class_label=True) # Klassifizierung der Datei
                #print(pred+" lid:", class_map.get(pred))
                # #prüfen ob HH, BD, Snare ansonsten nichts
                if(class_map.get(pred,"nix")=="nix"):
                    print(dirName+os.sep+file+" ist kein Drum-Sample") # Rückmeldung Instrument nicht erkannt
                    print("prediction ist: ", pred)
                else:
                    # Speicherung der Audiodatei und Label in der Datenbank
                    sdat1 = (os.path.abspath(dirName+os.sep+file), class_map.get(pred))#unkomment
                    mycursor.execute(sqlformula, sdat1)
                    #print(os.path.abspath(dirName+os.sep+file))
                
    mydb.commit()
    print("done: Save Classified Data in DB")
    #return 'tabelle kommt noch Ordner: '+pfad
    return redirect(url_for('gebeTabAus'))# Aufruf der Antwort Webseite mit der Klassifikationstabelle durch die Methode gebeTabAus



# Auswertung der Formular der HTML-Seite mit der Angabe des Quellpfads
@app.route('/sortiererDialog', methods = ['GET', 'POST'])
def upload_Ordner():
    if request.method == 'POST':        
        quellPfad = request.form['quellPfad']             
        return redirect(url_for('sortiereSound', soundOrdner = quellPfad))


if __name__ == '__main__':
   #app.debug = True
   #app.run()   
   app.run(debug = True)# Ausführen der Applikation
   
