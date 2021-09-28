# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:41:37 2020

@author: Lucas & Caro
"""


import streamlit as st
import base64

from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from rdkit.ML.Scoring import Scoring
from sklearn.metrics import confusion_matrix
import math
import pandas as pd
import statistics
import plotly as py
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='LIDEB Tools - Metrics',
    layout='wide')

######
# Function to put a picture as header   
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

from PIL import Image
image = Image.open('cropped-header.png')
st.image(image)


st.markdown("![Twitter Follow](https://img.shields.io/twitter/follow/LIDeB_UNLP?style=social)")
st.subheader(":pushpin:" "About Us")
st.markdown("We are a drug discovery team with an interest in the development of publicly available open-source customizable cheminformatics tools to be used in computer-assisted drug discovery. We belong to the Laboratory of Bioactive Research and Development (LIDeB) of the National University of La Plata (UNLP), Argentina. Our research group is focused on computer-guided drug repurposing and rational discovery of new drug candidates to treat epilepsy and neglected tropical diseases.")
st.markdown(":computer:""**Web Site** " "<https://lideb.biol.unlp.edu.ar>")

# Introduction
#---------------------------------#

st.write("""
# LIDeB Tools - Metrics
**Metrics to evaluate the performance of the classificatory models **

Web App to evaluate the perfomance of classificatory models by calculation of AUCROC, BEDROC, Accuracy, F-measure, PR and EF.

The tool uses the following packages [RDKIT](https://www.rdkit.org/docs/index.html), [Scikit-learn](https://scikit-learn.org/stable/), [Plotly](https://plotly.com/python/)
    
""")


# image = Image.open('clustering_workflow.png')
# st.image(image, caption='Clustering Workflow')

#st.sidebar.header('Molecular descriptors')
st.sidebar.header('Upload your SMILES')

uploaded_file_1 = st.sidebar.file_uploader("Upload a TXT file with one SMILES per line", type=["txt"])
st.sidebar.markdown("""
[Example TXT input file](https://raw.githubusercontent.com/cbellera/Metrics/main/example.txt)
""")

st.sidebar.header('BEDROC')    
alpha_bedroc = st.sidebar.slider('Alpha BEDROC', 5, 100, 20, 5)

st.sidebar.header('ENRICHMENT FACTOR')    
fraccion_enriquesimiento = st.sidebar.slider('Fraction enrichment', 0.0, 0.95, 0.01 , 0.01)

st.sidebar.header('STANDARD DEVIATION')    
repeticiones = st.sidebar.slider('Number of iterations', 100, 1000, 100 , 50)
particion = st.sidebar.slider('Fraction of evaluation', 0.5, 0.95, 0.85 , 0.05)

st.sidebar.header('CONFUSION MATRIX')    
threshold_cm = st.sidebar.number_input('Threshold')



st.sidebar.title(":speech_balloon: Contact Us")
st.sidebar.info(
"""
If you are looking to contact us, please
[:e-mail:](mailto:lideb@biol.unlp.edu.ar) or [Twitter](https://twitter.com/LIDeB_UNLP)
""")



def metrics_calculation(uploaded_file_1):
    # Leemos el archivo cargado
    scores1 = pd.read_csv(uploaded_file_1, sep='\t', delimiter=None, header='infer', names=None)
    # Lo ordenamos
    scores1.sort_values(by=["score"],axis = 0, ascending = False,inplace = True, na_position ='last')
    
    y_modelo = list(scores1["score"])
    y_real = list(scores1["class"])
    
    clases_predichas = []
    for value in y_modelo:
        if value > threshold_cm:
            clases_predichas.append(1)
        else:
            clases_predichas.append(0)
            
    clases_predichas_serie = pd.Series(clases_predichas)
    # result = scores1["class"].eq(clases_predichas_serie)
    # proporcion_bc = round(sum(result) / len(result),4) # Numero de compuestos bien clasificados / numero de compuestos del dataset
    
    st.markdown(":point_down: **Here you can find the % of good classifications with the selected threshold**", unsafe_allow_html=True)
    # st.write(proporcion_bc)
    
    metric1 = round(accuracy_score(y_real, clases_predichas_serie),4)
    metric2 = round(balanced_accuracy_score(y_real, clases_predichas_serie),4)
    metric3 = round(average_precision_score(y_real, clases_predichas_serie),4)
    metric4 = round(f1_score(y_real, clases_predichas_serie),4)
    metric5 = round(precision_score(y_real, clases_predichas_serie),4)
    metric6 = round(recall_score(y_real, clases_predichas_serie),4)
    metric7 = round(jaccard_score(y_real, clases_predichas_serie),4)
    
    results = [metric1,metric2,metric3,metric4,metric5,metric6,metric7]
    metric_name = ["accuracy_score","balanced_accuracy_score","average_precision_score","f1_score","precision_score","recall_score","jaccard_score"]
    table_metrics = pd.Series(results,index=metric_name,name="Value")
    st.write(table_metrics)
   
    st.markdown(":point_down: **Here you can find the confusion matrix for the selected threshold**", unsafe_allow_html=True)
    matrix = pd.DataFrame(confusion_matrix(y_real, clases_predichas))
    matrix=matrix.rename(index={0: 'Predicted Positive', 1:"Predicted Negative"},columns={0:"Real Positive",1:"Real Negative"})

    st.write(matrix)
    
    # Hacemos los calculos para X repeticiones
    i=0   
    aucs=[]
    bedrocs=[]
    efs=[]
    prs=[]
    aupr_max= 1
    
    while i <= repeticiones:    # Para hacer el cross validation y poder sacar la desviacion del auc
    
        scores_modelo, X_test, clases_real, y_test = train_test_split(y_modelo, y_real, test_size= 1 - particion, random_state=i,stratify = y_real)
            
        array = tuple(zip(scores_modelo,clases_real))
        array_1 = pd.DataFrame(array)
        array_1.sort_values(0, axis = 0, ascending = False,inplace = True, na_position ='last')
        array_metricas = tuple(zip(array_1[0],array_1[1]))
        
        auc = Scoring.CalcAUC(scores= array_metricas, col= 1)   
        aucs.append(auc)
        bedroc = Scoring.CalcBEDROC(scores= array_metricas, col= 1,alpha=alpha_bedroc)
        bedrocs.append(bedroc)
        enrichment = Scoring.CalcEnrichment(scores= array_metricas, col= 1,fractions=[fraccion_enriquesimiento])
        efs.append(enrichment[0])
        auc_PR = average_precision_score(clases_real, scores_modelo)                        # Calculo el AUPR
        prs.append(auc_PR)
        
        i = i + 1
        
    
    # # Esto ese para el EFmax #
    n_total = len(y_real)           # TOT(tot)
    fraccion_ef = math.ceil(n_total*fraccion_enriquesimiento) # devuelve el numero entero mayor (TOT(1%))
    n_activos = sum(y_real)     # ACT(TOT)
    if fraccion_ef > n_activos:
        efmax = (n_activos/fraccion_ef)/(n_activos/n_total)
    else:
        efmax = 1/(n_activos/n_total)
    
    # Sacamos las medias y las desviaciones
    auc_ok= round(statistics.mean(aucs),4)
    sd_auc = round(statistics.stdev(aucs),4)
    
    bedroc_ok= round(statistics.mean(bedrocs),4)
    sd_bedroc = round(statistics.stdev(bedrocs)   ,4)
    
    ef_ok= round(statistics.mean(efs),4)
    sd_ef = round(statistics.stdev(efs),4)
    
    pr_ok= round(statistics.mean(prs),4)
    sd_pr = round(statistics.stdev(prs),4)
    
    resultado = [auc_ok,bedroc_ok,ef_ok, round(efmax,4), pr_ok,aupr_max]
    desviaciones = [sd_auc,sd_bedroc, sd_ef, 0, sd_pr, 0]
    
    final = pd.DataFrame([resultado,desviaciones]).T
    
    final=final.rename(index={0: 'AUC', 1:"BEDROC_" + str(alpha_bedroc), 2: 'EF_' + str(fraccion_enriquesimiento), 3:'EFmax_' + str(fraccion_enriquesimiento), 4:'AUC_PR', 5:"AUC_PRmax"},columns={0:"Mean",1:"SD"})
    
    st.markdown(":point_down: **Here you can see the calculated metrics**", unsafe_allow_html=True)

    st.write(final)
    
    return scores1, auc_ok, sd_auc,pr_ok, sd_pr, y_modelo, y_real

def roc_aupr_curves(auc_ok, sd_auc, pr_ok, sd_pr , y_modelo, y_real):

    title1= "ROC CURVE " + " " + "(" + "AUCROC = " + str(format(auc_ok, '.4f')) + " ± " + str(format(sd_auc, '.4f')) + ")"
    title2="PR CURVE " + " " + "(" + "AUCROC = " + str(format(pr_ok, '.4f')) + " ± " + str(format(sd_pr, '.4f')) + ")"
    fig = make_subplots(rows=1, cols=2, subplot_titles=(title1,title2))
    
    
    # # CURVA ROC
    fpr, tpr, thresholds = roc_curve(y_score= y_modelo,y_true= y_real)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, fill='tozeroy', showlegend=False),row=1, col=1)
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1,row=1, col=1)
    fig.update_xaxes(title_text="False Positive Rate (FPR)", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate (TPR)", row=1, col=1)

    
    # # CURVA PRECISION RECALL
    precision, recall, thresholds = precision_recall_curve(y_true= y_real, probas_pred = y_modelo)
    fig.add_trace(go.Scatter(x=recall, y=precision, fill='tozeroy', showlegend=False),row=1, col=2)
    fig.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=1, y1=0,row=1, col=2)
    
    fig.update_layout(margin = dict(t=60,r=20,b=20,l=20), autosize = True)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    st.plotly_chart(fig)


def ppv_plot(scores1):  
# importamos los paquetes necesarios:
    # scores = pd.read_csv(scores1, sep='\t', delimiter=None, header='infer', names=None)  
    y_modelo = list(scores1["score"])
    y_real = list(scores1["class"])
    # Determino la sensibilidad, especificidad y los valores de corte
    # fpr, tpr, thresholds = roc_curve(y_score= y_modelo, y_true= y_real,drop_intermediate=True)
    fpr, tpr, thresholds = roc_curve(y_score= y_modelo, y_true= y_real,drop_intermediate=False)  
    se = tpr
    sp = 1 - fpr
    se_sp = se/sp 
     
    tabla_final = pd.DataFrame(list(zip(thresholds,fpr,tpr,se,sp,se_sp)), 
                   columns =['Cutoff', 'fpr',"tpr","Se","Sp","Se/Sp"]) 
    
    # Calculos para agregar el valor de PPV a dos Ya distintos a la tabla
    
    prev1 = [0.001,0.005,0.01]
    
    se1=tabla_final["Se"].tolist()
    sp1=tabla_final["Sp"].tolist()
    
    tamanio = list(range(1, len(se1), 1))
    
    ppv_ok1 =[]
    
    for prev in prev1:
        ppv1a=[]
        for i in tamanio:
            ppv1 = (se1[i]*prev)/(se1[i]*prev + (1 - sp1[i])*(1-prev))
            ppv1a.append(ppv1)
        ppv_ok1.append(ppv1a)    
      
    # ppv_para_dos_ya = pd.DataFrame(ppv_ok1).T
    
    tabla_final_1 = pd.DataFrame(list(zip(thresholds,se,sp,se_sp,ppv_ok1[0],ppv_ok1[1],ppv_ok1[2])), 
                    columns =['Cutoff', "Se","Sp","Se/Sp","PPV (Ya = 0.001)","PPV (Ya = 0.005)","PPV (Ya = 0.01)"]) 
    
    
    # preparacion de datos para el grafico de  ppv
    
    prevalencia = pd.DataFrame(np.arange(0.00000000001, 0.012, 0.001))
    ya=prevalencia[0].tolist()
    
    se1=tabla_final["Se"].tolist()
    sp1=tabla_final["Sp"].tolist()
    
    tamanio = list(range(1, len(se1), 1))
    
    ppv_ok =[]
    
    for x in ya:
        ppv1=[]
        for i in tamanio:
            ppv = (se1[i]*x)/(se1[i]*x + (1 - sp1[i])*(1-x))
            ppv1.append(ppv)
        ppv_ok.append(ppv1)    
      
    ppv_tabla = pd.DataFrame(ppv_ok).T
    
    
    # GRAFICO PPV
    
    fig = go.Figure(go.Surface(opacity=1, 
        contours = {
            "x": {"show": False, "start": 0, "end": 0.01, "size": 0.04, "color":"white"},
            "y": {"show": False, "start": 0, "end": 2, "size": 0.04, "color":"white"},
            "z": {"show": False, "start": 0, "end": 1, "size": 0.2,"color":"white"}
        },
        x = ya,
        y = se_sp,
        z = ppv_tabla))
    
    fig.update_layout(
          scene = dict(
            xaxis = dict(title="Ya", nticks=10, range=[0,0.011],gridwidth = 10,linewidth = 10,
                         backgroundcolor="rgb(200, 200, 230)", gridcolor="white", showbackground=True, zerolinecolor="white"),
            yaxis = dict(title="Se/Sp",nticks=10, range=[0,2],gridwidth = 10,linewidth = 10,
                         backgroundcolor="rgb(230, 200,230)",
                            gridcolor="white",
                            showbackground=True,
                            zerolinecolor="white"),
            zaxis = dict(title="PPV",nticks=10, range=[0,1],gridwidth = 10,linewidth = 10,
                         backgroundcolor="rgb(230, 230,200)",
                            gridcolor="white",
                            showbackground=True,
                            zerolinecolor="white"),),
        margin=dict(r=20, l=10, b=10, t=10),
        font=dict(
            size=16,
            color="black")
        )
    st.plotly_chart(fig)
    
    st.markdown(":point_down: **Here you can see the table with cutoffs, Se, Sp and PPVs**", unsafe_allow_html=True)
    st.write(tabla_final_1)





if uploaded_file_1 is not None:
    scores1, auc_ok, sd_auc, pr_ok, sd_pr , y_modelo, y_real = metrics_calculation(uploaded_file_1)
    st.write("")
    st.markdown(":point_down: **Here you can see the ROC and AUPR curves**", unsafe_allow_html=True)
    roc_aupr_curves(auc_ok, sd_auc, pr_ok, sd_pr , y_modelo, y_real)
    
    st.markdown(":point_down: **Here you can see the PPV surface**", unsafe_allow_html=True)
    ppv_plot(scores1)
    
else:
    st.info('Awaiting for TXT file to be uploaded.')
 
   
    
#Footer edit

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made in  🐍 and <img style='display: ; ' href="https://streamlit.io" src="https://i.imgur.com/iIOA6kU.png" target="_blank"></img> Developed with ❤️ by <a style='display: ; text-align: center' href="https://twitter.com/capigol" target="_blank">Lucas Alberca</a> and <a style='display: ; text-align: center' href="https://twitter.com/carobellera" target="_blank">Caro Bellera</a> for <a style='display:; text-align: center;' href="https://lideb.biol.unlp.edu.ar/" target="_blank">LIDeB</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)


    