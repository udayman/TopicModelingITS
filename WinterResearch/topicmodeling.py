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
import pandas as pd
import numpy as np
import mglearn
import pyLDAvis
import pyLDAvis.sklearn

#data collection function, still suffers from minor typos due to the 'ignore' from writing to file
def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text



#script

lone = ''
directory = Path('/Users/Udayan Mandal/Documents/TopicModelingPapers/Machine Learning Models')
for file in directory.glob('**/*.pdf'):
    lone += convert_pdf_to_txt(str(file))
    f=open('xxx.txt','w',encoding='utf-8')
    f.write(lone)
    f.close()


with open('xxx.txt', 'r',encoding='utf-8') as f:
    clean_cont = f.read().splitlines()
    
    
shear=[i.replace('\xe2\x80\x9c','') for i in clean_cont ]
shear=[i.replace('\xe2\x80\x9d','') for i in shear ]
shear=[i.replace('\xe2\x80\x99s','') for i in shear ]

shears = [x for x in shear if x != ' ']
shearss = [x for x in shears if x != '']

dubby=[re.sub("[^a-zA-Z]+", " ", s) for s in shearss]

#Topic Modeling Part
vect=CountVectorizer(ngram_range=(1,1),stop_words='english')
dtm=vect.fit_transform(dubby)
pd.DataFrame(dtm.toarray(),columns=vect.get_feature_names())
lda = LatentDirichletAllocation(n_components=5)
lda_dtf = lda.fit_transform(dtm)
sorting=np.argsort(lda.components_)[:,::-1]
features=np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=range(5), feature_names=features,
sorting=sorting, topics_per_chunk=5, n_words=10)
Agreement_Topic=np.argsort(lda_dtf[:,2])[::-1]

zit=pyLDAvis.sklearn.prepare(lda,dtm,vect)
pyLDAvis.show(zit)


