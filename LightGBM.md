
# Light GBM Model Using Bayesian HyperParameter Tuning Plus Model Evaluation and Interpretation

### Data Prep


```python
##Read in Libraries
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot
import sklearn
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt.pyll
from hyperopt.pyll import scope
import sklearn.inspection
import shap

##Set Working Directory
os.chdir()

##Read in Training Data
Train = pd.read_csv('train.csv', delimiter=',', index_col=None)

##Read in Validation Data
Test = pd.read_csv('test.csv', delimiter=',', index_col=None)

##View Data
print(Train.shape)
print(Train['target'].value_counts())
Train.head()
```

### Data Preprocessing Function 


```python
#Impute, Oversample, Encode Categorical Variables and Create Model Inputs for Data

#Returns Final Training and Validation Sets along with Train and Test matrices to input into the model

def preprocessing(Train, Test, target):
    
    #List of Categorical Variables
    cat = Train.select_dtypes(include=['object', 'int64']).columns.tolist()
    
    #List of Continuous Variables
    cont = Train.select_dtypes(include=['float64']).columns.tolist()

    ##Imputing Data
    #Categorical - Set to 'Missing'/-1
    Train = Train[cat].fillna(-1)
    
    #Numerical - Impute Median of Column
    Train = Train[cont].fillna(Train.median())
    
    ##Over Sampling
    Train_1 = Train[Train[target]==1]
    #Change to Fit Desired Proportion of 1s to 0s
    Train_over = Train.append([Train_1]*10, ignore_index=True)
    print(Train_over[target].value_counts())
    print(Train_over.shape)

    ##Label Encoding Ordinal Variables
    ordinal = []
    #Training
    for feat in ordinal:
        le=sklearn.preprocessing.LabelEncoder()
    Train_over[feat] = le.fit_transform(Train_over[feat])
    #Validation
    for feat in ordinal:
        le=sklearn.preprocessing.LabelEncoder()
        Test[feat] = le.fit_transform(Test[feat])

    ##One Hot Encoding
    #Training
    Train_over=pd.get_dummies(Train_over, drop_first=True)
    #Validation
    Test=pd.get_dummies(Test, drop_first=True)

    ##Converting Data to Type Int for Model
    Train_ova[Train_ova.select_dtypes(['uint8']).columns]=Train_ova.select_dtypes(['uint8']).apply(lambda x: x.astype(float))
    Test[Test.select_dtypes(['uint8']).columns]=Test.select_dtypes(['uint8']).apply(lambda x: x.astype(float))
    
    ##Creating Matrices
    #Training Inputs
    X_train=Train_over.drop(columns=[target])
    #Validation Inputs
    X_test=Test.drop(columns=[target])
    #Training Target
    y_train=Train_over[target]
    #Validation Target
    y_test=Test[target]
    
    #Cleaning Column Names
    X_train.columns=["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns]
    X_test.columns=["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_test.columns]

    ##Output
    return Train_over, Test, X_train, X_test, y_train, y_test

```

### Building Light GBM Model Using Bayesian HyperParameter Tuning


```python
###HyperParameter Tuning Function
def hyperopt(X_train, X_test, y_train, y_test, param_space, num_eval):
    
    ##Setting HyperParamter Grid
    param_hyperopt={
        'learning rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
        'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 5, 100, 1)),
        'num_leaves': scope.int(hp.quniform('num_leaves', 5, 50, 1)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 1.0),
        'boosting_type': 'gbdt',
    }


    ##Defining Objective Function for Tuning
    def objective_function(params):

        #Evaluating LightGBM Classification Model on Tuning Parameters
        clf=lgb.LGBMClassifier(**params)

        evaluation = [(X_train, y_train), (X_test, y_test)]
            
        #Training Model
        clf.fit(X_train, y_train,
                eval_set=evaluation, eval_metric='auc',
                early_stopping_rounds=10, verbose=False)

        #Score Model on Validation to Obtain Predicted Probabilities
        preds=clf.predict_proba(X_test)
        preds=preds[:,1]

        #Adjusting Intercept of Predictions to Account for Oversampling Bias
        #Change to fit Target Proportion in Training and Oversampled Training
        newpreds=(preds * 0.8 * 0.02)/((1-preds) * 0.2 * 0.98 + preds * 0.8 * 0.02)

        #Evaluate Model and Adjust Hyperparamters to Maximize AUC
        auc=roc_auc_score(y_test, newpreds)

        print('Score:', auc)
        return {'loss': -auc, 'status': STATUS_OK}

    trials=Trials()

    #Parameter Tuning
    best_param=fmin(objective_function,
                        param_space,
                        algo=tpe.suggest,
                        max_evals=num_eval,
                        trials=trials,
                        rstate=np.random.RandomState(1))

    return best_param

##Running function to get best HyperParamters
best=hyperopt(X_train, X_test, y_train, y_test, param_hyperopt, 20)

best_param_values=[x for x in best.values()]
```

### Model Evaluation


```python
###Function to Build Model on Identified Best HyperParameters and Evaluate Model Performance
def evaluate(X_train, X_test, y_train, y_test, best_param_values):
    
    #Identify Best Parameters
    clf_best = lgb.LGBMClassifier(learning_rate = best_param_values[0],
                                 max_depth = best_param_values[1],
                                 n_estimators = best_param_values[2],
                                 num_leaves = best_param_values[3],
                                 colsample_bytree = best_param_values[4],
                                 bagging_fraction = best_param_values[5],
                                 boosting_type = 'gbdt')
    #Build Model on Parameters
    clf_best.fit(X_train, y_train)
    
    #Score Test Data
    preds=clf_best.predict_proba(X_test)
    preds=preds[:,1]
    #Adjust Predictions
    #Change to fit Target Proportion in Training and Oversampled Training
    newpreds=(preds * 0.8 * 0.02)/((1-preds) * 0.2 * 0.98 + preds * 0.8 * 0.02)
    
    #AUC on Validation
    auc=roc_auc_score(y_test, newpreds)
    
    #Track PPV and TPR Across Different Cutoffs
    cutoff=[]
    PPV=[]
    TPR=[]
    
    for i in np.arange(0, 1, 0.01):
        #Classifying Observations
        event=np.where(newpreds > i, 1, 0)
        #Positive Predicted Values
        ppv=precision_score(y_test, event, average='binary')
        #Confusion Matrix
        cm=confusion_matrix(y_test, event)
        tp=cm[1][1]
        fn=cm[1][0]
        fp=cm[0][1]
        tn=cm[0][0]
        #True Positive Rate
        tpr=tp/(tp+fn)
        #Appending Lists
        cutoff.append(i)
        PPV.append(ppv)
        TPR.append(TPR)
     
    #DataFrame of Cutoff Stats at Each Cutoff
    cutoff_stats = pd.DataFrame(data={'cutoff': cutoff, 'PPV': PPV, 'TPR': TPR})
    
    return cutoff_stats
    
evaluation = evaluate(X_train, X_test, y_train, y_test, best_param_values)

evaluation
```


```python
###Plot Model Stats (TPR and PPV) Across Different Cutoffs
pyplot.plot(cutoff, PPV, color='red')
pyplot.plot(cutoff, TPR, color='blue')
pyplot.show()
```


```python
###Confusion Matrix for Chosen Cutoff
cm = confusion_matrix(y_test, newpreds > #fill-in with chosen cutoff)
print(cm)
```

### Model Interpretation 


```python
###Feature Importance Plot
lgb.plot_importance(clf_best, max_num_features=20)
```


```python
###Change Data Type to Integer
X_train=X_train.apply(lambda x: x.astype(int))
X_test=X_test.apply(lambda x: x.astype(int))
```


```python
###SHAP Summary Plot
#SHAP Value (impact on model output) for different levels of key variables
shap_value=shap.TreeExplainer(clf_best).shap_values(X_train)
shap_value=np.array(shap_value[1])
shap.summary_plot(shap_value, X_train)
```


```python
###SHAP Explainer, Individual Observation
#For each individual observation, how each variable contributed to the final predicted probability
shap.initjs()
explainer=shap.SamplingExplainer(lambda x: clf_best.predict_proba(x)[:,1], data=X_train)

#Apply Explainer to Observation 100
shap_values=explainer.shap_values(X_train.loc[100,:])

#Plot of SHAP Values for Individual Prediction
shap.force_plot(explainer.expected_value, shap_values, features=X_train.loc[100,:],
               feature_names=X_train.columns.tolist())

```
