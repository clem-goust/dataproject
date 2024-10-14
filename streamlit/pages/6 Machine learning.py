## Initialisation

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import datetime as dt
import warnings
import streamlit as st

warnings.simplefilter('ignore')

## Importation des fonctions de préparation des données

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

## Importation des modèles

# Modèles adaptés à des petits jeux de données:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Modèle classique pour ce type de problèmes :
from sklearn.linear_model import LogisticRegression

# Modèles plus sophistiqués, non linéaires :
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

## Importation des metrics d'appréciation des modèles
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

# On n'importera pas GridSearchCV ici

# On fait aussi l'impasse, pour le moment, sur les fonctions de sélection des variables

## Chargement des fichiers

df_prospects_metrics = pd.read_csv(r"streamlit/output_streamlit/prospects_metrics.csv", index_col=0)

# Titre de la page

st.title("Garanteo | Machine Learning")

# Variables numériques:

df_prospects_num = df_prospects_metrics[[
    'age',
    'session_count', 
    'avg_days_btw', 
    'sum_duration_in_min',
    'CTC',
    'click',
    'sessions_before_first_ctc',
    'clicks_before_first_ctc',
    'time_online_before_first_ctc_in_sec', 
    'days_to_first_ctc'
]]

df_prospects_num = df_prospects_num.astype(float)

# Variables catégorielles:

df_prospects_cat = df_prospects_metrics[[
    'gender',
    'device_type',
    'device_browser',
    'device_operating_system',
    'device_language',
    'country',
    'user_preference', 
    'user_behavior'
]]

df_prospects_cat = df_prospects_cat.astype('category')

# On passe les variables numériques au MinMaxScaler

scaler_minmax = MinMaxScaler()
df_prospects_num_norm = pd.DataFrame(scaler_minmax.fit_transform(df_prospects_num), columns=df_prospects_num.columns)

# On passe les variables catégorielles au get_dummies

df_prospects_cat_norm = pd.get_dummies(df_prospects_cat)

# On regroupe le tout

df_prospects_Xnorm = pd.concat(axis = 1, objs = [
                                            df_prospects_metrics['user_id'],
                                            df_prospects_metrics['is_presented_prospect'],
                                            df_prospects_metrics['is_client'],
                                            df_prospects_num_norm, 
                                            df_prospects_cat_norm
                                                ])

# On se focalise sur le Groupe 1 (groupe d'entraînement)

# Groupe 1 = prospects déjà appelés par les sales
# On extrait les variables d'entrainement
df_group_1_Xnorm = df_prospects_Xnorm[df_prospects_Xnorm['is_presented_prospect']==1]\
                                    .drop(columns=['user_id','is_presented_prospect', 'is_client'])

# On extrait la variable à prédire
df_group_1_y = df_prospects_metrics[df_prospects_metrics['is_presented_prospect']==1]['is_client']

# Groupe 2 = groupe dont on veut prédire la proba de devenir client s'ils sont appelés
# On étudie donc les prospects pas encore clients ET pas encore appelés
# On extrait les variables
df_group2_Xnorm = df_prospects_Xnorm[(df_prospects_Xnorm['is_presented_prospect']==0) & (df_prospects_Xnorm['is_client']==0)]\
                        .drop(columns=['user_id', 'is_presented_prospect', 'is_client'])


models = {
    "K Nearest Neighbors": 
        KNeighborsClassifier(
            n_neighbors = 3, 
            p = 1, 
            weights = 'uniform'
        )
    , 
    "Gaussian Naive Bayes": 
        GaussianNB(var_smoothing= 1e-09)
    , 
    "Support Vector Machine":
        SVC(
            probability = True,
            C = 0.1,
            degree = 2,
            gamma = 'scale',
            kernel = 'linear'
        )
    , 
    "Logistic Regression":
        LogisticRegression()
    , 
    "Random Forest":
        RandomForestClassifier(
            bootstrap = True, 
            max_depth = 10, 
            max_features = 'sqrt', 
            min_samples_leaf = 1, 
            min_samples_split = 2, 
            n_estimators = 100
        )
    ,
    "Gradient Boosting":
        GradientBoostingClassifier(
            learning_rate = 0.1, 
            max_depth = 3, 
            max_features = None, 
            min_samples_leaf = 1, 
            min_samples_split = 2, 
            n_estimators = 100, 
            subsample = 1.0
        )
}

descriptions = {
    "K Nearest Neighbors":{
        "description":"The KNeighborsClassifier implements learning based on a simple majority vote of the k nearest neighbors of each query point, where k is an integer value specified by the user.",
        "pros":'''
            - Very simple model
            - Decent predictions
        ''',
        "cons":'''
            - Not really designed for probability prediction
            - Becomes less performant as the number of variables increase
            - Optimum reached for k=3, meaning that lead scores will fall in max. 4 categories (5 wanted)
        ''',
        "image":"knn",
        "source":"Source: scikit-learn.org"
    },
    "Gaussian Naive Bayes":{
        "description":"Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes' theorem with the 'naive' assumption of conditional independence between every pair of features given the value of the class variable.",
        "pros":'''
            - Very simple model 
        ''',
        "cons":'''
            - Variable independence is a strong assumption, even if we removed highly-correlated variables
            - Makes poor predictions (see Classification report and Confusion Matrix)
        ''',
        "image":"nb",
        "source":"Source: scikit-learn.org"
    },
    "Support Vector Machine":{
        "description":"A support vector machine constructs a hyper-plane or set of hyper-planes which can be used for classification, regression or other tasks. Intuitively, a good separation is achieved by the hyper-plane that has the largest distance to the nearest training data points of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier.",
        "pros":'''
            - Effective with a high number of variables
            - Good handling of non-linear relationships
        ''',
        "cons":'''
            - Not really designed for probability predictions
            - May be complex to adjust
        ''',
        "image":"svm",
        "source":"Source: scikit-learn.org"
    },
    "Logistic Regression":{
        "description":"The logistic regression model assigns probabilities to the possible outcomes using a logistic function. As an optimization problem, the logistic function of a binary-class logistic regression (such as ours: the called lead converts into client / (s)he does not) is calibrated by minimizing a cost function.",
        "pros":'''
            - The typical model for probability estimation in the context of a binary classification
            - Quite easy to interpret
        ''',
        "cons":'''
            - Linear model, potentially not very efficient in case of no linear / direct relationships between variables
            - Highly dependent on regularisation, which leads to disappointing results in our case
        ''',
        "image":"lr",
        "source":"Source for text: scikit-learn.org | Source for image: wikipedia.org"
    },
    "Random Forest":{
        "description":"Decision trees create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.",
        "pros":'''
            - Good results for binary classification
            - Deemed more powerful than linear models, notably to grasp complex relationships between variables
            - Automatically assigns importances to variables
        ''',
        "cons":'''
            - Designed for larger samples (risk of overfitting)
        ''',
        "image":"rf",
        "source":"Source for text: scikit-learn.org | Source for image: spotfire.com"
    },
    "Gradient Boosting":{
        "description":"Gradient Boosting regressors are additive models, computing the sum of numerous decision tree regressors called _weak learners_ (models that are only slightly better than random guessing, such as small decision trees). The process is iterative, and each newly added tree is fitted in order to minimize a sum of losses given the previous ensemble.\n\n In case of a classification, the assignment of a class is not direct, as each of the _weak learners_ predict continuous values. The mapping from the value to a class or a probability depends on a loss function.",
        "pros":'''
            - Deemed more powerful than linear models, notably to grasp complex relationships between variables
            - Automatically assigns importances to variables 
        ''',
        "cons":'''
            - May be complex to adjust
            - Designed for larger samples (risk of overfitting) 
            - More resource-intensive (not a big issue here given the small size of the dataset)
        ''',
        "image":"gb",
        "source":"Source for text: scikit-learn.org | Source for image: linkedin.com @Pratik Thorat"
    }
}

model = st.sidebar.selectbox("Select a classification model", list(models.keys()))

with st.expander(label="## Rationale for using machine learning in Project Garanteo"):
    st.write(f'''
            ##### To help sales prioritizing their calls, Garanteo asked for a scoring model that would evaluate the conversion potential of each lead.\n
            - On the one hand, Garanteo has around 1,272 leads for which _conversion likelihood should be predicted_\n
            - On the other hand Garanteo also has 655 prospects that were already called by the sales team, some of which became clients (while others did not): we have _historical conversion data_\n
            - Besides, Garanteo has meaningful information about his prospects, among which connection setup, trafic on website, personal data: we have _explanatory variables_ \n
            - Last but not least, independence testing showed that none of these datapoints have direct and strong influence on client conversion: the _correlation is difficult to grasp_\n
            '''
    )
    with st.container(border=True):
        st.write("**For all these reasons, training a machine learning model seemed particularly appropriate.**")

@st.cache_resource(max_entries=10)
def train_models(_model_dict=models):
    
    # On applique train_test_split au Groupe 1 :
    X_train, X_test, y_train, y_test = train_test_split(df_group_1_Xnorm, df_group_1_y, train_size = 0.80, random_state=11)

    # On crée un dictionnaire de résultats :
    result_dict={
        "K Nearest Neighbors":{
            "trained_model":"",
            "y_pred":"",
            "y_proba":"",
            "y_proba_g2":""
        }, 
        "Gaussian Naive Bayes":{
            "trained_model":"",
            "y_pred":"",
            "y_proba":"",
            "y_proba_g2":""
        }, 
        "Support Vector Machine":{
            "trained_model":"",
            "y_pred":"",
            "y_proba":"",
            "y_proba_g2":""
        }, 
        "Logistic Regression":{
            "trained_model":"",
            "y_pred":"",
            "y_proba":"",
            "y_proba_g2":""
        }, 
        "Random Forest":{
            "trained_model":"",
            "y_pred":"",
            "y_proba":"",
            "y_proba_g2":""
        },
        "Gradient Boosting":{
            "trained_model":"",
            "y_pred":"",
            "y_proba":"",
            "y_proba_g2":""
        }
    }

    # On entraîne tous les modèles et on stocke leurs prédictions
    for key in _model_dict.keys():
        try:
            model = _model_dict[key].clone()
        except:
            model = _model_dict[key]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        result_dict[key]['trained_model']=model
        result_dict[key]['y_pred']=y_pred
        result_dict[key]['y_proba']=y_proba
        result_dict[key]['y_proba_g2']= model.predict_proba(df_group2_Xnorm)

    # On renvoie les résultats des modèles mais aussi du train_test_split
    return result_dict, X_train, X_test, y_train, y_test

result_dict, X_train, X_test, y_train, y_test = train_models()

tab_0, tab_1, tab_2, tab_3, tab_4 = st.tabs([
    ":robot_face: Model",
    ":star: Lead scores",
    ":game_die: Probabilities",
    ":memo: Classification report",
    ":dart: Calibration curve"
])

with tab_0:
    st.subheader(model)
    st.write(descriptions[model]['description'])
    col_a, col_b, col_c = st.columns([1,5,1])
    with col_a:
        st.write()
    with col_b:
        st.image(f"streamlit/images/{descriptions[model]['image']}.png", caption=descriptions[model]['source'], use_column_width=True)
    with col_c:
        st.write()
    st.subheader(f":white_check_mark: Pros:")
    st.write(descriptions[model]['pros'])
    st.subheader(f":x: Cons:")
    st.write(descriptions[model]['cons'])
    
with tab_1:
    df_scoring = df_prospects_metrics[(df_prospects_metrics['is_presented_prospect']==0)\
                                    & (df_prospects_metrics['is_client']==0)]['user_id'].to_frame().reset_index(drop=True)

    df_scoring['proba'] = pd.DataFrame(result_dict[model]['y_proba_g2'])[1]
    df_scoring['score'] = pd.cut(df_scoring['proba'], bins=5, labels=[1,2,3,4,5])

    g = sns.catplot(
        data=df_scoring, 
        x='score',
        kind='count',
        height=7,
        aspect=1.7,
        color='steelblue'
    )

    g.set_axis_labels("\nScore", f"Count\n", fontsize=20)
    g.set_xticklabels(labels=[1,2,3,4,5], fontsize=20)
    g.set_yticklabels(fontsize=20)

    for p in g.ax.patches:
        g.ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),  # Position de l'étiquette
                ha='center', va='center', 
                xytext=(0, 10),  # Décalage pour placer l'étiquette au-dessus de la barre
                textcoords='offset points', 
                fontsize=15, color='black')

    st.subheader(f"Lead scores distribution")
    st.pyplot(g)

with tab_2:
    fig, ax = plt.subplots()
    sns.histplot(
        data=pd.DataFrame(result_dict[model]['y_proba_g2'])[1],
        binwidth=0.01,
        shrink = 0.8,
        ax=ax, 
        color='steelblue',
        edgecolor=None
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set(xlabel="Probability", ylabel=f"Count")
    st.subheader(f"Probabilities distribution")
    st.pyplot(fig)

with tab_3:
    st.subheader("Report")
    report = pd.DataFrame(classification_report(y_test, result_dict[model]['y_pred'], output_dict=True))\
        .rename(lambda x: x.capitalize(), axis=0)\
        .rename(lambda x: x.capitalize(), axis=1)
    report = ((report*100).astype(int).astype(str)+"%").drop('Support')
    report['Accuracy']['Precision']=''
    report['Accuracy']['Recall']=''
    st.dataframe(report, use_container_width=True)
    with st.expander("See explanations"):
        st.markdown("##### Precision Score")
        st.write("The precision score is the ability of the classifier not to label as positive a sample that is negative.")
        st.latex(r"Precision = \frac{True\ positives}{True\ positives + False\ positives}")
        st.divider()
        st.markdown("##### Recall Score")
        st.write("The recall score is the ability of the classifier to find all the positive samples.")
        st.latex(r"Recall = \frac{True\ positives}{True\ positives + False\ negatives}")
        st.divider()
        st.markdown("##### Accuracy Score")
        st.write("The accuracy score is the ability of the classifier to label correctly.")
        st.latex(r"Accuracy = \frac{True\ positives + True\ negatives}{Total\ population}")
        st.divider()
        st.markdown("##### F1-Score")
        st.write("The F1 score can be interpreted as a harmonic mean of the precision and recall")
        st.latex(r"F1 = \frac{2 * True\ positives}{2 * True\ positives + False\ positives + False\ negatives}")
        st.divider()
    
    st.subheader("Confusion matrix")
    matrix = confusion_matrix(y_test, result_dict[model]['y_pred'])
    col1a, col1b = st.columns(2)
    col1a.metric(":green[**True negative**]", matrix[0][0])
    col1a.metric(":red[**False negative**]", matrix[1][0])
    col1b.metric(":red[**False positive**]", matrix[0][1])
    col1b.metric(":green[**True positive**]", matrix[1][1])

    st.subheader("Area Under ROC Curve")
    st.metric("AUC-ROC", round(roc_auc_score(y_test, result_dict[model]['y_pred']),2))
    with st.expander("See explanations"):
        st.markdown("##### Receiver Operating Characteristic (ROC) Curve")
        st.write("A receiver operating characteristic (ROC) curve is a graphical plot which illustrates the performance of a binary classifier system as its discrimination threshold is varied. It is created by plotting the fraction of true positives out of the positives (~ precision) vs. the fraction of false positives out of the negatives, at various threshold settings")
        st.divider()
        st.markdown("##### Area Under ROC Curve")
        st.write("By computing the area the curve information is summarized in one number. A perfect model has an area of 1 (only true positives, no false negatives). A model with an AUC-ROC of 0.5 is likely to be purely random")
        st.divider()

with tab_4:
    prob_true, prob_pred = calibration_curve(y_test, result_dict[model]['y_proba'][:,1], n_bins=10)
    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker='o')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Share of positives')
    st.subheader(f'Calibration curve')
    with st.expander("See explanations"):
        st.markdown("##### Calibration curve")
        st.write("Well calibrated classifiers are probabilistic classifiers for which the predicted probabilities can be directly interpreted as a confidence level. For instance, a well calibrated (binary) classifier should classify the samples such that among the samples to which it gave a probability value close to, say, 0.8, approximately 80% actually belong to the positive class.")
    st.markdown(f'#### {model}')
    st.pyplot(fig)

with st.expander(label="Model selection"):
    st.write(f'''
            As seen in the classification reports, all models struggle to predict customer conversions (likely a dataset issue: too small, too random). AUC-ROC around 0.5 indicates that the models' outputs remain quite random. That said:\n
            - We can rule out *Gaussian Naive Bayes* as it gives unsatisfactory results (numerous prediction errors).\n
            - We can rule out *SVM* because the probabilities obtained are poorly discriminated.\n
            *Logistic Regression* and *KNearestNeighborsClassifier (with k=3)* give interesting results, but:\n
            - By design, *KNN with k=3* only produces a maximum of 4 probability levels. While these levels are well discriminated, too few prospects stand out to achieve meaningful scoring. Additionally, it sometimes makes some errors.\n
            - *Logistic Regression* is a linear regression model. Yet, preliminary analyses did not demonstrate any linear or direct relationship between the explanatory variables and customer conversion.\n
            Thus, *Gradient Boosting Classifier* and *Random Forest Classifier* remain. Although unable to predict customer conversions, both obtain average probabilities close to the historical proportion of client conversions (around 7-8%). They also have the advantage of not being linear models.\n
            - However, *Gradient Boosting Classifier* seems more dependent on its parameter settings, which seems risky given such a small training sample. It also produces false positives, which is not the case with *Random Forest Classifier*.\n
            '''
    )
    with st.container(border=True):
        st.write("**Therefore, for illustration purposes (and for illustration only, as I observe that the results are not very satisfactory), we used a trained Random Forest Classifier model for the Leads Dashboard.**")
