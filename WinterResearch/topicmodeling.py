from __future__ import  print_function
from pathlib import Path
import re
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy
import pandas as pd
import numpy as np
import mglearn
import pyLDAvis
import pyLDAvis.sklearn

from sklearn.model_selection import GridSearchCV

#data collection function, still suffers from minor typos due to the 'ignore' from writing to file
def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-16'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    try:
        fp = open(path, 'rb')
    except:
        device.close()
        retstr.close()
        return ''
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    try:
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
            interpreter.process_page(page)
    except:
        fp.close()
        device.close()
        retstr.close()
        return ''
        

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text



#script

'''
lone = ''
#directory = Path('/Users/Udayan Mandal/Documents/TopicModelingPapersTest')
#directory = Path('/Users/Udayan Mandal/Documents/TopicModelingPapers/Machine Learning Models')
directory = Path('/Users/Udayan Mandal/Documents/TopicModelingPapersAll')

#directory = Path('/Users/Udayan Mandal/Documents/paratransitpapers')
f=open('xxx.txt','w',encoding='utf-16')
for file in directory.glob('**/*.pdf'):
    next_document = False
    lone = convert_pdf_to_txt(str(file))
    for line in lone.split('\n'):
        if (line.lower() == 'references'):
            f.write("NEW DOCUMENT START UNIQUE")
            next_document = True
            break
        if ('IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS') in line:
            continue
        line = line.replace('ﬁ', 'fi')
        line = line.replace('ﬂ', 'fl')
        line = line + ' '
        f.write(line)
    if not next_document:
        f.write("NEW DOCUMENT START UNIQUE")
f.close()
'''


with open('xxx.txt', 'r',encoding='utf-16') as f:
    clean_cont = f.read().split("NEW DOCUMENT START UNIQUE")

clean_cont = [i for i in clean_cont if i != ' ']
print('# Documents :', len(clean_cont))
shear =[i.replace('\xe2\x80\x9c','') for i in clean_cont ]
shear =[i.replace('\xe2\x80\x9d','') for i in shear ]
shear =[i.replace('\xe2\x80\x99s','') for i in shear ]

shear = [x for x in shear if x != ' ']
shear = [x for x in shear if x != '']
dubby =[re.sub("[^a-zA-Z]+", " ", s) for s in shear]

not_allowed = ['fig', 'cid', 'vol', 'et', 'al', 'was', 'has', 'this', 'use', 'datum', 'set', 'model', 'algorithm']

nlp = spacy.load('en', disable=['parser', 'ner'])
for document in range(len(dubby)):
    new_words = []
    doc = nlp(dubby[document])
    lemmatized_out = [token.lemma_ if token.lemma_ != "-PRON-" else "" for token in doc]
    for word in lemmatized_out:
        if (word.lower() not in not_allowed):
            new_words.append(word)
    dubby[document] = ' '.join(new_words)

    
#Topic Modeling Part

#custom key words here

'''
key_words = "LIDAR point Weakly supervised learning Data Visualization Graphical user Interfaces Interactive Systems Traffic light recognition Traffic signals Object detection Computer Vision Intelligent Transportation System Active Safety R-CNN Kalman Filter Multimodal data fusion Surveillance Vehicle Tracking Driving Data Long-term prediction Non-parametric Bayes Autonomous Vehicle High dynamic range imaging Deep learning Traffic forecasting Spatial-temporal Graph convulution LSTM Radar Tracking Variational methods Sensor fusion Taxi system Demand forecast Big transportation data Vehicle detection SURF (Symmetric Speeded Up Robust Features) MMR (Vehicle make and model recognition) Anomaly Detection hidden Markov models human dynamics Dimensionality reduction Graph embedding Sparse representation Transportation mode recognition Cellular phone sensor data Urban computing ML learning algorithms heirarchical modeling Fuzzy logic systems set-membership computational complexity reduction adaptive algorithms Internet of vehicles Deep reinforcement learning Road detection off-road environments vision-based method traversable region detection fundamental mask multi-scale classification online learning dynamic dataset transportation ridesharing quality of experience online social network services recommendation systems mobile applications Stereo matching cost unsupervised training unlabeled data intelligent surveillance vehicle classification intelligent vehicle on-road vehicle detection occlusion handling machine learning methods driving behaviors prediction state and intent recognition advanced driver assistance systems taxi destination prediciton support vector recognition ensemble learning stress level prediction stress friendly driving behavior stress level classification point to set visual tracking varying viewpoint handling pattern recognition tracking filters TASC(train automatic stop control) online learning algorithm reinforcement learning urban metro systems smart traffic management concept drift unsupervised incremental learning impact propogation traffic optimization traffic control social media analytics identification models in-vehicle data recorders classification stacked generalization supervised learning feature extraction high-level fusion ROI extraction image enhancement collaborative filtering technique highway traffic-flow prediction object tracking decision tree digital road driving events driving situations linear logistic regression mining appearance patterns multi-orientation detection orientation estimation air traffic control predictive models gaussian mixture model clustering methods vehicle and pedestrian detection multipart model synthetic training data time series forecasting taxi demand semantic information topic modeling deep gaussian processes driveability metric public dataset risk assessment traffic hazards road safety vulnerable road users movement modeling intention recognition motion classification trajectory prediction speed profile prediction neural networks traffic data trip modeling high-speed train positioning error LSSVM ant colony optimization K-means algorithm location error online sparse optimization iterative pruning error minimization L0-norm minimization decision making prediction making prediction methods public transit system multi-agent systems path planning on-vehicle experience simulation passenger behavior modeling time-expanded graph traffic flow prediction sensitivity to loss of data traffic density vehicle counting smart cards public transportation clustering analysis city structure data analysis data mining spatio-temporal medium-term mobility prediction pattern mining atypical travel pattern multi-airport systems traffic flow pattern driving analytics driving behavior bag of words insurance telematics social media event identification subway passenger flow prediction social sensing transit ridership georeference jaccard distance placename toponym traffic tweet vernacular geography natural language processing geographic information science CNN meta learning smartphone travel surveys GPS trajectories mode inference active learning transport simulation simulation metamodels gaussian processes eco-driving driving style driving evalutation driving assistance system driver sleepiness feature selection real driving danger-level analysis driving risks multisourced driving data driver behavior driver warning systems intenton prediction robust control car-following intersection traffic light deep deterministic policy gradient traffic oscillation driving agent driving distraction euclidean distance fuzzy logic fuzzy neural networks vehicle safety social transportation traffic information detection text mining accident detection stacked autoencoder artificial intelligence discrete event simulation distributed event simulation distributed decision-making travel demand prediciotn origin-destination prediction multi-scale convolutional NSTM network origin-destination tensor survey road extraction remote sensing image local dirchlet mixture model multisical-high-order intelligent vehicle video surveillance computational intelligence artificial neural networks convulution model-combination unsupervised clustering generative models DSRC non-line-of-sight V2V safety communication reliable safety beacons road transportation trajectory data literature review visual analytics big data aggressive driving connected vehicle data horizontal curves random forest traffic safety data-driven ITS unlicensed taxi driving conditions hybrid electric vehicle intelligent multifeature statistical discrimination statistical feature traffic sign detection tracking recognition incremental learning speed prediction evolving fuzzy neural network K-means clustering remote traffic microwave sensors arterial road networks hierarchical temporal memory driving data analysis driving safety verification drivers similarity model checking graph-based analysis active safety system analytic heirarchy process ridge regression model bisecting k-means clustering extreme learning machine generalization ability pdhf robustness vehicle positioning mobility sensing flow multi source data collection travel survey gps tracking data processing trip ends identification data-driven method traffic mode detection trip purpose prediction knowledge discovery context-aware applications automated machine learning fine grained classification cascading recognition deformable part models support vector machines acoustic signal processing land vehicles accident prevention driver distraction and inattention intelligent supporting systems OLWSVR short-term traffic flow forecast supervised algorithm Fast fourier transform railway wagons regression algorithm vertical acceleration context-free grammars intelligent transportation system applications video analysis general adverserial network autonomous vehicle control cooperative adaptive cruise control policy gradient algorithms lifetime optimization linear programming semi markov model transportation system statistical learning pattern analysis railway safety railway accidents wavelet transforms naturalistic riding study powered two wheelers"
key_words = key_words.lower()
key_words = key_words.replace('-','')
key_vocab = key_words.split(' ')

key_vocab = [word for word in set(key_vocab)]


vect=CountVectorizer(ngram_range=(1,1),stop_words='english',vocabulary=key_vocab)
'''


vect=CountVectorizer(ngram_range=(1,2), max_df = 0.75, lowercase=True, stop_words='english')
dtm=vect.fit_transform(dubby)
#pd.DataFrame(dtm.toarray(),columns=vect.get_feature_names())

'''
search_params = {
  'n_components': [5, 10, 15, 20, 25, 30],
  'learning_decay': [.5, .7]
}

# Set up LDA with the options we'll keep static
model = LatentDirichletAllocation()

# Try all of the options
gridsearch = GridSearchCV(model, param_grid=search_params, n_jobs=-1, verbose=1)
gridsearch.fit(dtm)

# What did we find?
print("Best Model's Params: ", gridsearch.best_params_)
print("Best Log Likelihood Score: ", gridsearch.best_score_)
'''


lda = LatentDirichletAllocation(n_components=5)
lda.fit(dtm)
#lda_dtf = lda.fit_transform(dtm)
sorting=np.argsort(lda.components_)[:,::-1]
features=np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=range(5), feature_names=features,
sorting=sorting, topics_per_chunk=5, n_words=30) #feature names here
#Agreement_Topic=np.argsort(lda_dtf[:,2])[::-1]
zit=pyLDAvis.sklearn.prepare(lda,dtm,vect)
pyLDAvis.show(zit)




