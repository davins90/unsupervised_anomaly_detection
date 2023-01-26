import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import prince
import gower
import heapq
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as pgo

from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import beta
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import *
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import FunctionTransformer
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score

pio.renderers.default='iframe'

imp_mean = IterativeImputer(random_state=0)
imp_mean2 = SimpleImputer(strategy='constant', fill_value='missing',verbose=0,add_indicator=True)
scaler = MinMaxScaler()

###

def partitioning(df,train_dim, val_test_dim,target):
    """
    
    """
    val_test_dim_edit = train_dim + val_test_dim
    train, val, test = np.split(df.sample(frac=1,random_state=2), [int(train_dim*len(df)), int(val_test_dim_edit*len(df))])
    print("Train shape: ",train.shape)
    print("Train %: \n",train[target].value_counts(normalize=True))
    print("Val %: \n",val[target].value_counts(normalize=True))
    print("Test %: \n",test[target].value_counts(normalize=True))
    return train, val, test


def auc_score(y_trues, y_preds):
    """
    
    """
    for i, y_pred in enumerate(y_preds):
        y_true = y_trues[i]
        auc = roc_auc_score(y_true, y_pred)
    return auc

def adversarial_validation(val,test,drop_cols):
    """
    
    """
    aval = val.copy()
    atest = test.copy()
    # 1 - define target
    aval['y'] = 1.0
    atest['y'] = 0.0
    # 2 - create dataframe
    ad = aval.append(atest).sample(frac=1,random_state=2)
    # 3 - drop unuseful columns
    c_drop = drop_cols
    ad = ad.drop(columns=c_drop)
    # 4 - define format and imputation
    for i in ad:
        if ad[i].dtypes != 'object':
            ad[i] = ad[i].astype(float)
            ad[i] = ad[i].fillna(ad[i].median())
        else:
            ad[i] = ad[i].fillna('missing')
            ad[i] = ad[i].astype(str)
    # 5 - model preparation
    y = ad['y'].values
    X = ad.drop(columns=['y']).values
    categorical_features_indices = np.where(ad.dtypes != np.float64)[0]
    # 6 - train test split
    adv_X_train, adv_X_test, adv_y_train, adv_y_test = train_test_split(X, y , test_size = 0.30 , random_state = 2)
    train_data = Pool(data=adv_X_train,label=adv_y_train,cat_features=categorical_features_indices)
    test_data = Pool(data=adv_X_test,label=adv_y_test,cat_features=categorical_features_indices)
    # 7 - model training
    params = {'iterations': 1000,'eval_metric': 'AUC','od_type': 'Iter','od_wait': 50}
    model = CatBoostClassifier(**params)
    _ = model.fit(train_data, eval_set=test_data, plot=False, verbose=False)
    # 8 - model evaluation
    auc = auc_score([test_data.get_label()],[model.predict_proba(test_data)[:,1]])
    if auc <= 0.6:
        return print("No distribution shift, OK! AUC is: ",auc)
    else:
        return print("Check features importance (to be added) and rerun. AUC is: ",auc)
    
def class_imbalance(train,target):
    """
    
    """
    df_under = train[train[target]==train[target].value_counts(normalize=False).index[1]]
    df_over = train[train[target]==train[target].value_counts(normalize=False).index[0]]
    df_over = df_over.sample(frac=df_under.shape[0]/df_over.shape[0],random_state=2)
    final_train = df_over.append(df_under).sample(frac=1,random_state=2)
    print(final_train[target].value_counts(normalize=False))
    print("Train dataset shape: ",final_train.shape)
    return final_train

def compute_mca(model, df, n_components, n_iter, copy, check_input, engine, random_state):
    """
    
    """
    mca = model
    mca = model(n_components,n_iter,copy,check_input,engine,random_state).fit(df)
    print("MCA explained variance: ", mca.explained_inertia_)
    return mca

def plot_coordinates_plotly2d(model, X, x_component=0, y_component=1, show_column_points=True):
    """
    
    """
    fig = pgo.Figure()

    # Plot column principal coordinates
    if show_column_points or show_column_labels:

        col_coords = model.column_coordinates(X)
        x = col_coords[x_component]
        y = col_coords[y_component]

        prefixes = col_coords.index.str.split('_').map(lambda x: x[0])

        fig.add_traces([pgo.Scatter(
            x=x[prefixes == prefix], 
            y=y[prefixes == prefix], 
            hovertext=col_coords[prefixes == prefix].index, 
            mode='markers',name=prefix) for prefix in prefixes.unique()])

    # Text
    fig.update_layout(showlegend=True)
    ei = model.explained_inertia_
    fig.update_xaxes(title_text='Component {} ({:.2f}% inertia)'.format(x_component, 100 * ei[x_component]))
    fig.update_yaxes(title_text='Component {} ({:.2f}% inertia)'.format(y_component, 100 * ei[y_component]))

    return fig

def plot_coordinates_plotly3d(model, X, x_component=0, y_component=1, z_component=2, show_column_points=True):
    """
    3d
    """
    fig = pgo.Figure()

    # Plot column principal coordinates
    if show_column_points or show_column_labels:

        col_coords = model.column_coordinates(X)
        x = col_coords[x_component]
        y = col_coords[y_component]
        z = col_coords[z_component]

        prefixes = col_coords.index.str.split('_').map(lambda x: x[0])

        fig.add_traces([pgo.Scatter3d(
            x=x[prefixes == prefix], 
            y=y[prefixes == prefix],
            z=z[prefixes == prefix], 
            hovertext=col_coords[prefixes == prefix].index, 
            mode='markers',name=prefix) for prefix in prefixes.unique()])

    # Text
    fig.update_layout(showlegend=True)
    ei = model.explained_inertia_
    fig.update_xaxes(title_text='Component {} ({:.2f}% inertia)'.format(x_component, 100 * ei[x_component]))
    fig.update_yaxes(title_text='Component {} ({:.2f}% inertia)'.format(y_component, 100 * ei[y_component]))
    fig.update_yaxes(title_text='Component {} ({:.2f}% inertia)'.format(z_component, 100 * ei[z_component]))

    return fig

def log_transform(x):
    return np.log(x + 1)

def model_training(X_train, y_train, numerical_cols, categorical_cols,parameters,multiclass):
    """
    
    """
    
    ##
    transformer = FunctionTransformer(log_transform)
    numerical_preprocessor = Pipeline(steps=[("imputer", IterativeImputer(ExtraTreesRegressor(n_estimators=5,random_state=1),random_state=1,verbose=0,add_indicator=True)),
                                     ("scaler", transformer)]) #MinMaxScaler()
    categorical_preprocessor = Pipeline(steps=[("imputer", SimpleImputer(strategy='constant', fill_value='missing',add_indicator=True)),
                                           ("label_enc", OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[("numerical_preprocessor", numerical_preprocessor, numerical_cols),
                                               ("categorical_preprocessor", categorical_preprocessor, categorical_cols)])
    pipe_model = GradientBoostingClassifier(random_state=0,n_iter_no_change=25,warm_start=True,max_features=1.0)
    ##
    model = Pipeline(steps=[('preprocessor', preprocessor),('model', pipe_model)])
    
    ##
    # ('model', CalibratedClassifierCV(base_estimator=pipe_model,method='isotonic'))])
    ##
    if multiclass == 'no':
        model_grid = GridSearchCV(model,parameters,cv=4,scoring='recall',verbose=0,return_train_score=True).fit(X_train,y_train)
    else:
        model_grid = GridSearchCV(model,parameters,cv=4,scoring='accuracy',verbose=0,return_train_score=True).fit(X_train,y_train)
    print('GridSearchCV results...')
    print("Mean Train Scores: \n{}\n".format(model_grid.cv_results_['mean_train_score']))
    print("Mean CV Scores: \n{}\n".format(model_grid.cv_results_['mean_test_score']))
    print("Best Parameters: \n{}\n".format(model_grid.best_params_))
    
    return model_grid


def model_evaluation(model,X_test,y_test):
    """
    
    """
    print('Test results...')
    y_test_predict_grid = model.predict(X_test)    
    # print("Calibration: ")
    # x, y = calibration_curve(y_test_predict_grid, y_test, n_bins = 10, normalize = True)
    # plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')   
    # plt.plot(y, x, marker = '.', label = 'Decision Tree Classifier') 
    # plt.xlabel('Average Predicted Probability in each bin') 
    # plt.ylabel('Ratio of positives') 
    # plt.legend()
    # plt.show() 
    print("Model Test Recall:", metrics.recall_score(y_test, y_test_predict_grid))
    print('--------------------------------------------------')
    print('Model Test Confusion Matrix')
    cm = confusion_matrix(y_test,y_test_predict_grid,normalize='true') 
    cmd = ConfusionMatrixDisplay(cm,display_labels=['No','Yes'])
    cmd.plot()
    print('Classification report : \n',classification_report(y_test, y_test_predict_grid))

    ##
    # feature_importances = model.best_estimator_.named_steps['model'].feature_importances_
    feature_importances = model.named_steps['model'].feature_importances_
    feature_names = X_test.columns
    lista = []
    for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
            print('{}: {}'.format(name, score))
            lista.append(name)
    # print('First ten features by importances:')
    # print(lista[0:15])
    
    return model

def model_calibration(model,X_test,y_test):
    """
    
    """
    best_model = model.best_estimator_
    calibrator = CalibratedClassifierCV(best_model, cv='prefit')
    calibrator = calibrator.fit(X_test,y_test)
    return calibrator

def log_beta_scaling(df,name1, name2):
    """
    
    """
    names = [name1, name2]
    for i in df:
        a = 9
        b = 1
        if i in names:
            df[i+'_ris'] = np.log1p(df[i])
            df[i+'_ris'] = scaler.fit_transform(df[i+'_ris'].values.reshape(-1,1))
            df[i+'_ris'] = beta.ppf(df[i+'_ris'], a, b)
    return df, scaler

def log_beta_transform(df,scaler,name1, name2):
    """
    
    """
    names = [name1, name2]
    for i in df:
        a = 9
        b = 1
        if i in names:
            df[i+'_ris'] = np.log1p(df[i])
            df[i+'_ris'] = scaler.transform(df[i+'_ris'].values.reshape(-1,1))
            df[i+'_ris'] = beta.ppf(df[i+'_ris'], a, b)
    return df

def warning_score_prod(df, weight_f1=0.8):
    """
    
    """
    df['warning_score'] = 0
    w = [weight_f1,1 - weight_f1]
    df['warning_score'] = np.dot(df[['num_accounts_related_to_user_ris','num_days_previous_transaction_ris']],w)
    df['warning_score'] = df['warning_score'].mask(df['device_info_v4'] == 'other',df['warning_score']+0.05)
    df['warning_score'] = df['warning_score'].mask(df['browser_enc'] == 'other',df['warning_score']+0.1)
    df['warning_score'] = df['warning_score'].mask(df['warning_score']>=1,0.95)
    df = df.drop(columns=['num_accounts_related_to_user_ris','num_days_previous_transaction_ris'])
    df['warning_score'] = df['warning_score'].mask(df['num_accounts_related_to_user']<=1.0,df['warning_score']/2)
    return df



def warning_score_dev(df):
    """
    
    """
    df['warning_score'] = 0
    w = [0.8,0.2]
    df['warning_score'] = np.dot(df[['max_c_ris','max_d_ris']],w)
    df['warning_score'] = df['warning_score'].mask(df['device_info_v4'] == 'other',df['warning_score']+0.05)
    df['warning_score'] = df['warning_score'].mask(df['browser_enc'] == 'other',df['warning_score']+0.1)
    df['warning_score'] = df['warning_score'].mask(df['warning_score']>=1,0.95)
    df = df.drop(columns=['max_c_ris','max_d_ris'])
    df['warning_score'] = df['warning_score'].mask(df['max_c']<=1.0,df['warning_score']/2)
    return df

def beta_fusion(prior, like, w_prior):
    """
    http://doingbayesiandataanalysis.blogspot.com/2012/06/beta-distribution-parameterized-by-mode.html
    
    """
    maxk = 20
    mink = 2
    w_like = 1- w_prior
    k_prior = w_prior*(maxk-mink)+mink
    k_like = (1-w_prior)*(maxk-mink)+mink
    alfa_prior = prior*(k_prior-2)+1
    beta_prior = (1-prior)*(k_prior-2)+1
    alfa_like = like*(k_like-2)+1
    beta_like = (1-like)*(k_like-2)+1
    post = (alfa_prior+alfa_like-1)/(alfa_prior+alfa_like + beta_prior+beta_like-2)
    return post

def features_eng(df, version):
    """
    
    """
    if version == 'clustering':
        df = df.drop(columns=['dist2','TransactionID'])
    elif version == 'anomaly':
        df = df.drop(columns=['dist2','customer_id','TransactionID'])
    
    df = df.rename(columns={'id_31':'browser'})
    df['P_emaildomain'] = df['P_emaildomain'].mask(df['P_emaildomain']=='gmail','gmail.com')
    df['R_emaildomain'] = df['R_emaildomain'].mask(df['R_emaildomain']=='gmail','gmail.com')
    df['id_30'] = df['id_30'].replace(" ","_",regex=True)
    df['id_30'] = df['id_30'].str.replace(".","_",regex=False)
    df['browser'] = df['browser'].mask(df['browser'].str.contains('SM') | df['browser'].str.contains('ZTE'),'other')
    df['browser'] = df['browser'].astype(str)
    df['browser_enc'] = 'other'
    df['browser_enc'] = df['browser_enc'].mask(df['browser'].str.contains('ie'),'ie')
    df['browser_enc'] = df['browser_enc'].mask(df['browser'].str.contains('safari'),'safari')
    df['browser_enc'] = df['browser_enc'].mask(df['browser'].str.contains('edge'),'edge')
    df['browser_enc'] = df['browser_enc'].mask(df['browser'].str.contains('firefox'),'firefox')
    df['browser_enc'] = df['browser_enc'].mask(df['browser'].str.contains('android'),'android')
    df['browser_enc'] = df['browser_enc'].mask(df['browser'].str.contains('Android'),'android')
    df['browser_enc'] = df['browser_enc'].mask(df['browser'].str.contains('chrome'),'chrome')
    df['browser_enc'] = df['browser_enc'].mask(df['browser'].str.contains('opera'),'opera')
    df['browser_enc'] = df['browser_enc'].mask(df['browser'].str.contains('google'),'chrome')
    df['browser_enc'] = df['browser_enc'].mask(df['browser'].str.contains('samsung'),'android')
    df['DeviceType'] = df['DeviceType'].mask(df['browser']== 'ie 11.0 for tablet','tablet')
    df = df.drop(columns='browser')
    df['device_info'] = df['DeviceInfo'].replace("-","_",regex=True)
    df['device_info2'] = df['device_info'].replace(" ","_",regex=True)
    df['device_info'] = df['DeviceInfo'].replace("-","_",regex=True)
    df['device_info2'] = df['device_info'].replace(" ","_",regex=True)
    df['device_info3'] = df['device_info2'].str.split("_").str[0]
    df['device_info3'] = df['device_info3'].str.lower()
    df['device_info_v4'] = 'other'
    df['device_info_v4'] = df['device_info_v4'].mask(((df['device_info3']=='windows') | (df['device_info3']=='microsoft') | (df['device_info3']=='trident/7.0')),'windows')
    df['device_info_v4'] = df['device_info_v4'].mask(((df['device_info3']=='ios') | (df['device_info3']=='iphone')),'ios')
    df['device_info_v4'] = df['device_info_v4'].mask(df['device_info3']=='macos','ios')
    df['device_info_v4'] = df['device_info_v4'].mask(df['device_info3']=='blade','blade')
    df['device_info_v4'] = df['device_info_v4'].mask(df['device_info3']=='lenovo','lenovo')
    df['device_info_v4'] = df['device_info_v4'].mask(df['device_info3']=='redmi','redmi')
    df['device_info_v4'] = df['device_info_v4'].mask(df['device_info3']=='pixel','pixel')
    df['device_info_v4'] = df['device_info_v4'].mask(df['device_info3']=='android','android')
    df['device_info_v4'] = df['device_info_v4'].mask(df['device_info3']=='macos','ios')
    df['device_info_v4'] = df['device_info_v4'].mask(df['device_info3']=='alcatel','alcatel')
    df['device_info_v4'] = df['device_info_v4'].mask(df['device_info3']=='nokia','nokia')
    df['device_info_v4'] = df['device_info_v4'].mask(df['device_info3']=='asus','asus')
    df['device_info_v4'] = df['device_info_v4'].mask(df['device_info3']=='oneplus','oneplus')
    df['device_info_v4'] = df['device_info_v4'].mask(df['device_info3']=='zte','zte')
    df['device_info_v4'] = df['device_info_v4'].mask(df['device_info3']=='macos','ios')
    df['device_info_v4'] = df['device_info_v4'].mask(df['device_info3']=='hisense','hisense')
    df['device_info_v4'] = df['device_info_v4'].mask(df['device_info3']=='linux','linux')
    df['device_info_v4'] = df['device_info_v4'].mask((df['device_info3'].str.contains("lg") | (df['device_info3'].str.contains("nexus"))),'lg')
    df['device_info_v4'] = df['device_info_v4'].mask(((df['device_info3'].str.contains("huawei")) | (df['device_info3'].str.contains("hi6210sft"))),'huawei')
    df['device_info_v4'] = df['device_info_v4'].mask(((df['device_info3']=='motog3') |(df['device_info3']=='moto')),'moto')
    df['device_info_v4'] = df['device_info_v4'].mask(((df['device_info3']=='sm') |(df['device_info3']=='samsung')),'samsung')
    df = df.drop(columns=['device_info','device_info2','device_info3','DeviceInfo'])
    return df

### Clustering

def clustering_preparation(df,version):
    """
    
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    for i in df:
        if i in numerical_cols:
            if i != 'customer_id':
                df[i] = imp_mean.fit_transform(X = df[i].values.reshape(-1,1))
                df[i] = scaler.fit_transform(X = df[i].values.reshape(-1,1))
    for i in df:
        if i in categorical_cols:
            df[i] = df[i].astype(str)
            df[i] = imp_mean2.fit_transform(X = df[i].values.reshape(-1,1))
    if version == 'training':
        return df, imp_mean, imp_mean2, scaler
    elif version == 'prediction':
        return df
    
def clustering_encoding(df):
    """
    
    """
    # Product
    df['product_enc'] = 999
    df['product_enc'] = df['product_enc'].mask(df['ProductCD']=='C',0)
    df['product_enc'] = df['product_enc'].mask(df['ProductCD']=='H',1)
    df['product_enc'] = df['product_enc'].mask(df['ProductCD']=='S',2)
    df['product_enc'] = df['product_enc'].mask(df['ProductCD']=='R',3)
    df = df.drop(columns='ProductCD')
    # Card4
    df['card4_enc'] = 999
    df['card4_enc'] = df['card4_enc'].mask(df['card4']=='visa',0)
    df['card4_enc'] = df['card4_enc'].mask(df['card4']=='mastercard',1)
    df['card4_enc'] = df['card4_enc'].mask(df['card4']=='american express',2)
    df['card4_enc'] = df['card4_enc'].mask(df['card4']=='discover',3)
    df = df.drop(columns='card4')
    # Card6
    df['card6_enc'] = 999
    df['card6_enc'] = df['card6_enc'].mask(df['card6']=='debit',0)
    df['card6_enc'] = df['card6_enc'].mask(df['card6']=='credit',1)
    df = df.drop(columns='card6')
    # Device Type
    df['DeviceType_enc'] = 999
    df['DeviceType_enc'] = df['DeviceType_enc'].mask(df['DeviceType']=='mobile',0)
    df['DeviceType_enc'] = df['DeviceType_enc'].mask(df['DeviceType']=='desktop',1)
    df['DeviceType_enc'] = df['DeviceType_enc'].mask(df['DeviceType']=='tablet',2)
    df = df.drop(columns='DeviceType')
    # Browser
    df['browser_enc2'] = 7
    df['browser_enc2'] = df['browser_enc2'].mask(df['browser_enc']=='chrome',0)
    df['browser_enc2'] = df['browser_enc2'].mask(df['browser_enc']=='safari',1)
    df['browser_enc2'] = df['browser_enc2'].mask(df['browser_enc']=='firefox',2)
    df['browser_enc2'] = df['browser_enc2'].mask(df['browser_enc']=='ie',3)
    df['browser_enc2'] = df['browser_enc2'].mask(df['browser_enc']=='android',4)
    df['browser_enc2'] = df['browser_enc2'].mask(df['browser_enc']=='edge',5)
    df['browser_enc2'] = df['browser_enc2'].mask(df['browser_enc']=='opera',6)
    df = df.drop(columns='browser_enc')
    # Device info
    df['device_info_v4_enc'] = 17
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='ios',0)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='windows',1)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='samsung',2)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='hisense',3)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='moto',4)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='pixel',5)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='lg',6)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='blade',7)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='huawei',8)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='oneplus',9)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='alcatel',10)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='redmi',11)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='lenovo',12)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='asus',13)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='linux',14)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='android',15)
    df['device_info_v4_enc'] = df['device_info_v4_enc'].mask(df['device_info_v4']=='zte',16)
    df = df.drop(columns='device_info_v4')
    # Index
    df = df.set_index('customer_id')
    return df

def clustering_main(df,version,max_cluster,choose_n_cluster):
    """
    
    """
    if version == 'training':
        sil=[]
        dft = df.copy()
        for num_clusters in list(range(2,max_cluster)):
            kproto = KPrototypes(n_clusters=num_clusters, verbose=0,random_state = 301,n_init=5)
            cluster_labels = kproto.fit_predict(dft.values, categorical=[3,4,5,6,7,8])
            dft['cluster_labels'] = cluster_labels
            dist = gower.gower_matrix(dft.drop(columns=['cluster_labels']),cat_features = [False, False ,False, True, True,True,True,True,True])
            sil.append(silhouette_score(dist,labels=cluster_labels, metric='precomputed'))
            print('For cluster number: ',num_clusters,' the score is: ', sil[-1])
    elif version == 'choosen':
        kproto = KPrototypes(n_clusters=choose_n_cluster, verbose=0,random_state = 301)
        cluster_labels = kproto.fit_predict(df.values, categorical=[3,4,5,6,7,8])
        centroid = []
        for j,i in enumerate(kproto.cluster_centroids_):
            centroid.append(pd.DataFrame(i,index=['TransactionAmt','num_accounts_related_to_user', 'num_days_previous_transaction','product_enc', 'card4_enc', 'card6_enc',
               'DeviceType_enc', 'browser_enc2', 'device_info_v4_enc'],columns=[j]))
        centroid = pd.concat(centroid,axis=1)
        centroid.style.background_gradient(cmap='brg',axis=1)
        df['cluster_labels'] = cluster_labels
        print(df['cluster_labels'].value_counts())
        return df, centroid, kproto
    
def clustering_prediction(df,model,version):
    """
    
    """
    if version == "prod":
        df['cluster_labels_pred'] = model.predict(X=df[['TransactionAmt', 'num_accounts_related_to_user', 'num_days_previous_transaction',
                                                    'product_enc', 'card4_enc', 'card6_enc', 'DeviceType_enc','browser_enc2',
                                                    'device_info_v4_enc']],categorical=[3,4,5,6,7,8])
    else:
        df['cluster_labels_pred'] = model.predict(X=df[['TransactionAmt', 'max_c', 'max_d',
                                                    'product_enc', 'card4_enc', 'card6_enc', 'DeviceType_enc','browser_enc2',
                                                    'device_info_v4_enc']],categorical=[3,4,5,6,7,8])
    return df