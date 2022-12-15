import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def select_features(X_train, X_train_df, y_train, X_test, n):

    """
    Description
    -----------
    Select top n relevant features.
    
    Arguments
    ----------
    X_train: training set.
    X_train_df: dataframe of X_train.
    y_train: labels of training set.
    X_test: test set.
    n: number of features.

    Return
    ---------
    X_train_fs & X_test_fs: X_train and X_test with n selected features.
    l_feat & l_pos: list of the n features and their position. 
    """
    
    l_feat = []
    l_pos = []
    fs = SelectKBest(score_func = f_classif, k = n)
    fs.fit(X_train, y_train)
    index = fs.fit(X_train, y_train).get_support(indices = True)
    l_pos.append(index)
    l_feat.append(X_train_df.columns[:][index])

    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    
    return X_train_fs, X_test_fs, l_feat, l_pos


def metrics(pipeline, X, y):

    """
    Description
    -----------
    Gets Sensitivity and specificity.
  
    Arguments
    ----------
    pipeline: designed pipeline for a model.
    X: X_train or X_test.
    y: y_train or y_test.

    Return
    ---------
    Sensitivity and specificity values.
    """

    prediction = pipeline.predict(X)
    cm = confusion_matrix(y, prediction)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()

    # pathogenic = sensitivity
    sens = (confusion_matrix(y, prediction)[1][1] / 
           (confusion_matrix(y, prediction)[1][0]+
            confusion_matrix(y, prediction)[1][1])) * 100
            
    # benign = specificity
    spec = (confusion_matrix(y, prediction)[0][0] / 
           (confusion_matrix(y, prediction)[0][0]+
            confusion_matrix(y, prediction)[0][1])) * 100
    
    sens = round(sens, 2)
    spec = round(spec, 2)
    
    return sens, spec


def get_roc_auc_score(model, X_tr, y_tr, X_te, y_te, print_table = True):

    """
    Description
    -----------
    Get ROC-AUC score.
    
    Arguments
    ----------
    model: model to use.
    X_tr: training set.
    y_tr: labels of training set.
    X_te: test set.
    y_te: labels of test set.
    print_table=True (default).

    Return
    ---------
    Returns a table with ROC-AUC.
    for training and test sets.
    """
    
    y_tr_p = model.predict(X_tr)
    y_te_p = model.predict(X_te)
  
    er_tr = [roc_auc_score(y_tr, y_tr_p)]
    er_te = [roc_auc_score(y_te, y_te_p)]
    
    ers = [er_tr, er_te]
    headers=["Roc-auc"]
    
    if print_table:
        print("%10s" % "", end="")
    
        for h in headers:
            print("%10s" % h, end="")
        print("")
    
        headersc = ["Train", "Test"]
    
        cnt = 0
        for er in ers:
            hc = headersc[cnt]
            cnt = cnt + 1
            print("%10s" % hc, end="")
    
            for e in er:
                print("%10.2f" % e, end="")
            print("")
    
    return ers
    

def challenge_encoding(X_train, X_ch):
    
    """
    Description
    -----------
    Do a categorical encoding.
    
    Arguments
    ----------
    X_train (numpy array): training data with which we train the model.
    They are needed as a basis in the categorical encoding. 
    Can be the output of sklearn.train_test_split() function.
    
    X_ch (numpy array): data from which we want to predict its label.
        
    Returns
    ----------
    X_ch_enc (numpy array): encoded codified numpy array that models need.
    X_ch_df (DataFrame): encoded codified df. 
    
    """

    # Convert numpy arrays in df
    c = ['affected_domain', 'initial_aa', 'final_aa', 'functional_domain',
         'd_size', 'd_hf', 'd_vol', 'd_msa', 'd_charge', 'd_pol', 'd_aro', 
         'residue_conserv', 'secondary_str', 'plddt', 'str_landscape', 'MTR']
    
    X_train_df = pd.DataFrame(X_train, columns = c)
    X_ch_df = pd.DataFrame(X_ch, columns = c)
    
    l = ['affected_domain','initial_aa', 'final_aa', 'functional_domain',
         'd_charge', 'd_pol', 'd_aro', 'secondary_str', 'str_landscape']
    
    for i in l: 
        # First: LabelEncoder()
        le = LabelEncoder()
        le.fit(X_train_df[i]) # fit only in training set
        domain_labels_train = le.transform(X_train_df[i])
        domain_labels_ch = le.transform(X_ch_df[i])
        X_train_df["label_train"] = domain_labels_train
        X_ch_df["label_ch"] = domain_labels_ch

        # Second: OneHotEncoder
        oe = OneHotEncoder()
        oe.fit(X_train_df[[i]])
        domain_feature_arr_train = oe.transform(X_train_df[[i]]).toarray()
        domain_feature_arr_ch = oe.transform(X_ch_df[[i]]).toarray()

        domain_feature_labels = list(le.classes_)
        domain_features_train = pd.DataFrame(domain_feature_arr_train, columns = domain_feature_labels)
        domain_features_ch = pd.DataFrame(domain_feature_arr_ch, columns = domain_feature_labels)


        # Update df
        X_ch_df = pd.concat([X_ch_df, domain_features_ch], axis = 1)
        X_ch_df = X_ch_df.drop(["label_ch",i], axis =1)

    return X_ch_df.to_numpy(), X_ch_df
    
    
def MLeKCNQ2_prediction(X_train, X_train_df, y_train, X_test, X_ch_enc):
   
    """
    Description
    -----------
    Execute MLe-KCNQ2 model prediction over a challenge dataset.
    
    Arguments
    ----------
    X_train: X_train data (already preprocessed).
    X_train_df: X_train dataframe (already preprocessed).
    y_train: y_train data.
    X_test: X_test data (already preprocessed).
    X_ch_enc: challenge data (encoded with the same encoding scheme as X_train).
        
    Returns
    ----------
    KCNQ2e_y_ch_p: is the label prediction over the challenge dataset.
    KCNQ2eprob: is the probability of previous prediction.
    
    """
    # Select the most informative 45 features
    X_train_fs, X_test_fs, featEn, posEn = select_features(X_train,
    							     X_train_df,
    							     y_train,
    							     X_test,
    							     n = 45) 
    # Create and fit a pipeline for the model
    pipeline_ensemble_soft = Pipeline( [("scaler", StandardScaler()), \
		                            ("Ensemble_soft", 
		                             VotingClassifier(voting = "soft",
		                                              weights = [1,0.5,1.75],
		                                              estimators=[
		                                                                   
		                                                  ("logistic", LogisticRegression(solver = "saga",
		                                                                                  penalty = "l2",
		                                                                                  max_iter = 10000,
		                                                                                  class_weight = {0: 3, 1: 2},
		                                                                                  multi_class = "ovr",
		                                                                                  C = 2.91,
		                                                                                  random_state = 8)),
		                                                                   
		                                                  ("SVC", SVC(kernel = "linear", 
		                                                              class_weight= {0:1, 1:1},
		                                                              probability=True,
		                                                              decision_function_shape = "ovr",
		                                                              degree = 2,
		                                                              gamma = "auto",
		                                                              C = 1,
		                                                              random_state = 45)),
		                                                                   
		                                                  ("RF", RandomForestClassifier(max_depth = 3,
		                                                                                criterion = "log_loss",
		                                                                                max_features = "log2",
		                                                                                oob_score = False,
		                                                                                min_samples_split = 2, 
		                                                                                class_weight= {0:3, 1:1},
		                                                                                random_state = 45))]))])
    # Fit the model
    pipeline_ensemble_soft.fit(X_train_fs, y_train)
    
    # Apply in X_ch same feature selection as for ensemble algorithm
    X_train_fs, X_ch_fs, feat_pred, pos_pred = select_features(X_train,
		                                                       X_train_df, 
		                                                    	 y_train,
		                                                    	 X_ch_enc, 
		                                                    	 n = 45)
		                                                    	 
    # Make prediction with MLe-KCNQ2 algorithm
    KCNQ2e_y_ch_p = pipeline_ensemble_soft.predict(X_ch_fs)

    # View probabilities in prediction
    KCNQ2eprob = pipeline_ensemble_soft.predict_proba(X_ch_fs)
	
    return  KCNQ2e_y_ch_p, KCNQ2eprob
