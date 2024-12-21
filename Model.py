import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, LearningCurveDisplay
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, r2_score, classification_report, ConfusionMatrixDisplay, roc_auc_score, matthews_corrcoef, RocCurveDisplay
from matplotlib import pyplot as plt
from seaborn import heatmap
from scipy.stats import probplot    
import seaborn as sns
import math
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import xgboost as xgb
import statistics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve

col_names = ['Sex', 'Length', 'Diameter', 'Height','Whole Weight', 'Shucked Weight','Viscera Weight', 'Shell Weight','Rings']
df = pd.read_csv("abalone.csv", header = None, names = col_names)
df['Sex'].replace(['M', 'F','I'], [0, 1, 2], inplace=True)
cdf = df.copy()

def visualisation():

    # Univariate Analysis
    print("----------------------------------------")
    print("Data Analysis")
    print("----------------------------------------")
    
    for attribute in col_names:
        min_value = df[attribute].min()
        max_value = df[attribute].max()
        null_count = df[attribute].isnull().sum()
        print(f"Attribute: {attribute}")
        print(f"Minimum value: {min_value}")
        print(f"Maximum value: {max_value}")
        if null_count > 0:
            print(f"Number of null values: {null_count}")
        else:
            print("No null values for this attribute.")
        print("\n")
    
    abalone = cdf[cdf['Height'] < 0.4]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Distribution")
    fig1 = abalone.hist(figsize=(20, 10), bins=20, edgecolor='black')
    plt.savefig("data_distribution.png")

    # probability plot
    fig2 = plt.figure(figsize=(10, 4))
    ax = plt.subplot(1, 1, 1)
    probplot(abalone['Rings'].values, dist='norm', plot=ax)
    ax.get_lines()[0].set_markersize(3)
    plt.title('Probability Plot for Rings')
    plt.ylabel('Rings')
    plt.grid(axis='y')
    plt.savefig("prob_plot.png")

    # Multivariate Analysis

    plt.figure(figsize=(8, 6))
    corr = abalone.corr()
    _ = sns.heatmap(corr, annot=True)
    plt.title("Correlation Heatmap")
    plt.savefig("heatmap.png")

class_boundaries = [0, 7, 10, 15, float('inf')]
def map_to_class(age):
    for i, boundary in enumerate(class_boundaries):
        if age <= boundary:
            return f'Class {i}'

# Apply the mapping function to the 'age' column
df['Rings'] = df['Rings'].apply(map_to_class)

def traintest_split(rannum):
    X = df.drop('Rings', axis=1)
    y = df['Rings']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = rannum)
    return(X_train, X_test, y_train, y_test)

def gb_split(rannum):
    X = df.drop('Rings', axis=1)
    y = df['Rings']
    label_encoder = LabelEncoder()
    y_xgb = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_xgb, test_size = 0.3, random_state = rannum)
    return(X_train, X_test, y_train, y_test)

def confidence_interval(accuracy):
    mean_accuracy = np.mean(accuracy)
    std = statistics.stdev(accuracy)
    z = 1.96
    upper_bound = mean_accuracy + z * (std / math.sqrt(len(accuracy)))
    lower_bound = mean_accuracy - z * (std / math.sqrt(len(accuracy)))

    return(lower_bound,upper_bound)

def decision_tree():
    best_accuracy = 0
    best_model = None
    ginif1 = []
    entropyf1 = []
    depth = [2,3,4,4,5]
    accuracy_g = []
    m_c = []
    for i in range(0,5):
        X_train, X_test, y_train, y_test = traintest_split(i)
        clf_gini = DecisionTreeClassifier(criterion = 'gini', random_state = i, max_depth = depth[i], splitter = 'best')
        clf_gini.fit(X_train, y_train)
        y_pred_gini = clf_gini.predict(X_test)
        accuracy_gini = accuracy_score(y_test, y_pred_gini)
        accuracy_g.append(accuracy_gini)

        mc = matthews_corrcoef(y_test, y_pred_gini)
        m_c.append(mc)
        if accuracy_gini > best_accuracy:
            best_accuracy = accuracy_gini
            best_model = clf_gini
        
        f1_g = f1_score(y_test,y_pred_gini, average = 'macro')
        ginif1.append(f1_g)

    
    mean_accuracy = np.mean(accuracy_g)
    std = statistics.stdev(accuracy_g)
    conf =  confidence_interval(accuracy_g)

    print("----------------------------------------")
    print("Decisoin Tree with Gini as criterion.")
    print("----------------------------------------")
    print("Mean accuracy is:", round(mean_accuracy,3))
    print("Confidence intarval is: ",round(conf[0],3),"-",round(conf[1],3))
    print("Matthews Coefficient is:", round(np.mean(m_c),3))
    print("F_1 Score (Macro): ",round(np.mean(ginif1),3))
    print("Standard deviation of Accuracy is:",round(std,3))

    acc_ent = []
    for j in range(0, 5):
        X_train_e, X_test_e, y_train_e, y_test_e = traintest_split(j)
        clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=j, max_depth = depth[j], splitter='best')
        clf_entropy.fit(X_train_e, y_train_e)
        y_pred_entropy = clf_entropy.predict(X_test_e)
        accuracy_entropy = accuracy_score(y_test_e, y_pred_entropy)
        acc_ent.append(accuracy_entropy)
        if accuracy_entropy > best_accuracy:
            best_accuracy = accuracy_entropy
            best_model = clf_entropy
        
        f1_e = f1_score(y_test,y_pred_entropy, average = 'macro')
        entropyf1.append(f1_e)

    mean_accuracy_ent = np.mean(acc_ent)
    std_ent = statistics.stdev(acc_ent)
    conf_ent =  confidence_interval(acc_ent)
    mean_f1_ent = np.mean(f1_e)

    print("\n----------------------------------------")
    print("Decisoin Tree with Entropy as criterion.")
    print("----------------------------------------")
    print("Mean accuracy is:", round(mean_accuracy_ent,3))
    print("Confidence intarval is: ",round(conf_ent[0],3),"-",round(conf_ent[1],3))
    print("Matthews Coefficient is:", round(np.mean(m_c),3))
    print("F_1 Score (Macro): ",round(np.mean(mean_f1_ent),3))
    print("Standard deviation of Accuracy is:",round(std_ent,3))

    # Decision Tree plotting

    plt.figure(figsize=(60, 40))
    fig3 = tree.plot_tree(best_model, filled=True, rounded=True, fontsize=14)
    plt.tight_layout()
    plt.savefig('best_decision_tree.png')
    plt.show()

    plt.plot(range(1, 6), ginif1, marker='o', label='Gini')
    plt.plot(range(1, 6), entropyf1, marker='o', label='Entropy')
    plt.xlabel('Experiment Number')
    plt.ylabel('F1 Score (Macro)')
    plt.title('F1 Scores of Gini and Entropy in Decision Tree')
    plt.legend()
    plt.savefig("decision_tree.png") 


def pre_pruning():

    X_train, X_test, y_train, y_test = traintest_split(10)
    clf_prune = DecisionTreeClassifier(random_state = 5 , ccp_alpha=0.02)
    grid_param={"criterion":["gini","entropy"],
                "splitter":["best","random"],
                "max_depth":range(2,10,1),
                "min_samples_leaf":range(1,10,1),
                "min_samples_split":range(2,10,1) 
                }
    
    grid_search=GridSearchCV(estimator=clf_prune,param_grid=grid_param,scoring = 'accuracy',cv=5,n_jobs=-1)

    grid_search.fit(X_train,y_train)
    best_params = grid_search.best_params_    

    best_clf = DecisionTreeClassifier(criterion=best_params['criterion'],
                                      max_depth=best_params['max_depth'],
                                      min_samples_leaf=best_params['min_samples_leaf'],
                                      min_samples_split=best_params['min_samples_split'],
                                      splitter=best_params['splitter'])
    

    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)

    print("\n----------------------------------------")
    print("Result of Per-Pruining")
    print("----------------------------------------")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix for this experiment.")
    cm = confusion_matrix(y_test, y_pred, labels=best_clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=best_clf.classes_)
    disp.plot()
    plt.title("Pre-Pruining Confusion Matrix")
    plt.show()
    print("F1-Score for this model is:", round(f1_score(y_test, y_pred, average='macro'),3))
    print("Matthews Coefficient for this experiment is: ",round(matthews_corrcoef(y_test, y_pred),3))    

def random_forest():

    accuracy_scores = []
    max_trees = 11
    depth_values = [i for i in range(2,12)]
    best_accuracy = 0
    best_depth = 0
    label_encoder = LabelEncoder()
    accuracy_scores_f = []
    f1_score_f = []
    m_c = []

    for i in range(2,max_trees + 1):
        X_train, X_test, y_train, y_test = traintest_split(i)
        clf = RandomForestClassifier(n_estimators = 125, criterion= 'log_loss', ccp_alpha = 0.015, max_depth= i)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_depth = i 
    
    print("\n----------------------------------------")
    print("Result of Random Forest")
    print("----------------------------------------")

    plt.figure(figsize=(8, 5))
    fig4 = plt.plot(depth_values, accuracy_scores, marker='o')
    plt.title('Random Forest Accuracy by Depth')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()        

    for i in range(1,11):
        X_train_f, X_test_f, y_train_f, y_test_f = traintest_split(i)
        final_forest = RandomForestClassifier(n_estimators = 125, criterion= 'log_loss', max_depth = best_depth, ccp_alpha = 0.015)
        final_forest.fit(X_train_f, y_train_f)
        y_pred_f = final_forest.predict(X_test_f)
        accuracy_f = accuracy_score(y_test_f, y_pred_f)
        f1s = f1_score(y_test_f, y_pred_f,average='macro')
        accuracy_scores_f.append(accuracy_f)
        f1_score_f.append(f1s)
        mc = matthews_corrcoef(y_test_f, y_pred_f)
        m_c.append(mc)
        if i == 6:
            print("Confusion Matrix for experiment number 6.")
            cm = confusion_matrix(y_test, y_pred_f, labels=final_forest.classes_)
            plt.figure(figsize=(8, 5))
            fig5 = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=final_forest.classes_)
            fig5.plot()
            plt.title("Confusion Martix Random Forest")
            plt.savefig("random_forest_matrix.png")

    conf = confidence_interval(accuracy_scores_f)
    std = statistics.stdev(accuracy_scores_f)
    print("Mean accuracy is:", round(np.mean(accuracy_scores_f),3))
    print("Confidence intarval is: ",round(conf[0],3),"-",round(conf[1],3))
    print("Matthews Coefficient is:", round(np.mean(m_c),3))
    print("F_1 Score (Macro): ",round(np.mean(f1_score_f),3))
    print("Standard deviation of Accuracy is:",round(std,3))   

accuracy_scores_xgb = []

def extremegradientboost():
    
    num_experiments = 3

    min_trees = 3

    max_trees = 25

    trees_step = 1

    depth_values = [i for i in range(3,26)]

    f1_scores = []

    aauracy_print = []
    for i in range(num_experiments):
        accuracy_scores_experiment_xgb = []
        f1_score_experiment_xgb = []
        X_train, X_test, y_train, y_test = gb_split(i)
        for n_trees in range(min_trees, max_trees + 1, trees_step):

            clf = xgb.XGBClassifier(n_estimators=n_trees)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)


            acc = accuracy_score(y_test, y_pred)
            f1_sc = f1_score(y_test,y_pred, average = 'macro')
            f1_score_experiment_xgb.append(f1_sc)
            accuracy_scores_experiment_xgb.append(acc)

        

        if i == 2:

            print("\n----------------------------------------")
            print("Results for XGBoost:")
            print("----------------------------------------")

            print("Confusion Matrix for experiment number 4.")
            cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                            display_labels=clf.classes_)
            disp.plot()

            plt.show()
            mc = matthews_corrcoef(y_test, y_pred)
            train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
                clf, X_train, y_train, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10), return_times=True, scoring='f1_macro'
            )

            plt.figure()
            plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training score')
            plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='Cross-validation score')
            plt.title('Learning Curve')
            plt.xlabel('Training examples')
            plt.ylabel('F1-Score')
            plt.legend(loc='best')
            plt.savefig("xgb.png")

            
        f1_scores.append(f1_score_experiment_xgb)
        accuracy_scores_xgb.append(np.mean(accuracy_scores_experiment_xgb))
    
    conf = confidence_interval((accuracy_scores_xgb))
    mean_accuracy = np.mean(accuracy_scores_xgb)
    std = statistics.stdev(accuracy_scores_xgb)
    mean_f1 = np.mean(f1_scores)



    print("Mean Accuracy Score is: ", round(np.mean(mean_accuracy),3))
    print("95% confidence interval of accuracy is: ",round(conf[0],3),"-",round(conf[1],3))
    print("Mean F1-Score is:",round(np.mean(mean_f1),3))        
    print("Matthews Coefficient for this experiment is: ",round(mc,3))
    print("Standard Deviation is:",round(std,3))

accuracy_scores_gbt = []
def gradientboosting():
    f1_scores = []

    num_experiments = 3
    # Minimum number of trees
    min_trees = 3
    # Maximum number of trees
    max_trees = 25
    # Step for the number of trees
    trees_step = 1

    depth_values = [i for i in range(3,26)]
    

    for i in range(num_experiments):
        accuracy_scores_gbt_experiment = []
        f1_scores_gbt_experiment = []
        # Split the data with a different random state for each experiment
        X_train, X_test, y_train, y_test = traintest_split(i)
        
        # Gradient Boosting
        for n_trees in range(min_trees, max_trees + 1, trees_step):
            # Train the Gradient Boosting classifier
            clf_gbt = GradientBoostingClassifier(n_estimators=n_trees)
            clf_gbt.fit(X_train, y_train)
            
            # Make predictions on the test set for Gradient Boosting
            y_pred_gbt = clf_gbt.predict(X_test)
            
            # Calculate accuracy and store for each number of trees (Gradient Boosting)
            acc_gbt = accuracy_score(y_test, y_pred_gbt)
            accuracy_scores_gbt_experiment.append(acc_gbt)

            f1_sc = f1_score(y_test, y_pred_gbt, average='macro')
            f1_scores_gbt_experiment.append(f1_sc)
        
        if i == 2:

            print("\n----------------------------------------")
            print("Results for Gradient Boosting:")
            print("----------------------------------------") 
            print("Confusion Matrix for experiment number 5.")
            cm = confusion_matrix(y_test, y_pred_gbt, labels=clf_gbt.classes_)
            fig6 = ConfusionMatrixDisplay(confusion_matrix=cm,
                                            display_labels=clf_gbt.classes_)
            fig6.plot()
            plt.title("Gradient Boosting Confusion Matrix")
            plt.savefig("gb.png")
            
            
            mc = matthews_corrcoef(y_test, y_pred_gbt)
            
        accuracy_scores_gbt.append(np.mean(accuracy_scores_gbt_experiment))
        f1_scores.append(np.mean(f1_scores_gbt_experiment))
    
    conf = confidence_interval(accuracy_scores_gbt)
    mean_acc = np.mean(accuracy_scores_gbt)
    std = statistics.stdev(accuracy_scores_gbt)
    mean_f1 = np.mean(f1_scores)

   

    print("Mean Accuracy over 3 experiments: ", round(mean_acc,3))
    print("95% Confidence interval: ",round(conf[0],3),"-",round(conf[1]),3)
    print("Standard Deviation of accuracy:",round(std,3))
    print("Matthews Coefficient for this experiment is: ",round(mc,3))
    print("Mean F1-Score is: ",round(mean_f1,3))

neural = []
def neural_networks():
    num_experiments = 3

    accuracy_scores_nn = []

    f1_scores = []

    m_c = []

    for i in range(num_experiments):
        accuracy_scores_nn_experiment = []
        # Split the data with a different random state for each experiment
        X_train, X_test, y_train, y_test = traintest_split(i)
        
        # Neural Network (Multi-layer Perceptron - MLP) using Adam optimizer
        clf_nn = MLPClassifier(solver='adam', learning_rate = 'adaptive', max_iter = 500)
        clf_nn.fit(X_train, y_train)
        y_pred_nn = clf_nn.predict(X_test)
        acc_nn = accuracy_score(y_test, y_pred_nn)
        f1_nn = f1_score(y_test,y_pred_nn, average = 'macro')
        accuracy_scores_nn_experiment.append(acc_nn)

        f1_scores.append(f1_nn)
        accuracy_scores_nn.append(accuracy_scores_nn_experiment)
        neural.append(f1_nn)
        if i == 2:
            print("----------------------------------------")
            print("Results for Neural Network (MLP) with Adam optimizer")
            print("----------------------------------------")

            print("Confusion Matrix for experiment number 4.")
            cm = confusion_matrix(y_test, y_pred_nn, labels=clf_nn.classes_)
            fig7 = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=clf_nn.classes_)
            fig7.plot()
            plt.title("Neural Networks Confusion Matrix")
            plt.savefig("neural.png")

        mc = matthews_corrcoef(y_test, y_pred_nn)
        m_c.append(mc)

    conf = confidence_interval(neural)
    mean_accuracy = np.mean(neural)
    std = statistics.stdev(neural)
    mean_f1 = np.mean(f1_scores)

    print("Mean Accuracy:",round(mean_accuracy,3))
    print("95% Confidence interval: ",round(conf[0],3),'-',round(conf[1],3))
    print("Standard Deviation of Accuracy:",round(std,3))
    print("Mean F1-Score is:",round(mean_f1,3))
    print("Matthews Coefficient",round(np.mean(m_c),3))

def regressor():
    mse_dtr = []
    mse_rfr = []

    X_train, X_test, y_train, y_test = gb_split(21)
    tree_reg = DecisionTreeRegressor(max_depth=3)  # You can experiment with different hyperparameters
    tree_reg.fit(X_train, y_train)
    tree_mse = r2_score(y_test, tree_reg.predict(X_test))

    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_train)
    rf_mse = r2_score(y_test, rf_reg.predict(X_test))

    print("\n----------------------------------------")
    print("Results for Random Forest Regressor:")
    print("----------------------------------------")

    print(f"Decision Tree R-squared score: {round(tree_mse,3)}")
    print(f"Random Forest MSE: {round(rf_mse,3)}")

    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 20)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
    rf_random.fit(X_train, y_train)
    
    print("\n Selected hyper-parameter Values:")
    print(rf_random.best_params_)

    best_random = rf_random.best_estimator_
    rf_best = r2_score(y_test, best_random.predict(X_test))
    print("\nBest R2-Score:",rf_best) 

# Write a combined graph of 3 models.

def com_xgb_gb_nn():
    x = [1,2,3]
    fig9 = plt.figure(figsize=(10, 4))
    plt.plot(x, neural, marker='o', label = 'Neural Network')
    plt.plot(x, accuracy_scores_gbt, marker='o', label = 'Gradient Boosting')
    plt.plot(x, accuracy_scores_xgb, marker='o', label='XGBoost')
    # Adding labels and title
    plt.xlabel('Experiment Number')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.title('Comparative Accuracy Scores')
    plt.grid(True)
    plt.savefig("Accuracy_comparision.png")

def main():
    visualisation()
    decision_tree()
    pre_pruning()
    random_forest()
    extremegradientboost()
    gradientboosting()
    neural_networks()
    regressor()
    com_xgb_gb_nn()

if __name__ == "__main__":
    main()