from collections import OrderedDict
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from utils import plot_confusion_matrix,classification_report_csv
from sklearn.preprocessing import label_binarize, normalize
import xgboost as xgb
import pandas as pd
import math
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pydotplus

from logging import getLogger
logger = getLogger("debug")


class ML_Model(object):

    def __init__(self, name, classifier, param_grid=None):
        
        self.classifier = classifier
        self.name = name
        
        if self.name in ('XGBoost', 'Random Forest'):
            self.feature_ranking = True
            
        self.param_grid = param_grid
        self.tn = self.fp = self.fn = self.tp = -1
        self.metrics = pd.DataFrame()
        self.vpn_metrics = OrderedDict()
        
        self.estimator = None
        self.model_path = None
        self.training_error = None
        self.is_trained = False

        self.score = None
        self.y_pred = None
        self.LABEL2DIG = {'chat':0, 'voip':1, 'trap2p':2, 'stream':3, 'file_trans':4, 'email':5 ,'vpn_chat':6, 'vpn_voip':7, 'vpn_trap2p':8, 'vpn_stream':9, 'vpn_file_trans':10, 'vpn_email':11}
        self.DIG2LABEL = {v: k for k, v in self.LABEL2DIG.items()}
    

    def train(self, X_train, y_train, random=False):
        if self.param_grid is None:
            print('----****-----')
            self.estimator = self.classifier
            self.classifier.fit(X_train, y_train)
            self.is_trained = True
            return
        elif random is False:
            print('in grid searchCV------->>>>\n')
            self.classifier = GridSearchCV(self.classifier, self.param_grid, cv=10, scoring='precision', n_jobs=-1)  # Do a 10-fold cross validation
            print('----->>>', self.classifier)
            self.estimator = self.classifier.estimator
            search_result = self.estimator.fit(X_train, y_train)
            self.is_trained = True
            return
        elif random is True:
            self.classifier = RandomizedSearchCV(self.classifier, param_distributions=self.param_grid,
                                                 n_iter=10, scoring='precision',
                                                 n_jobs=-1, cv=10, verbose=3, random_state=1001)
            #self.best_param = self.classifier.best_params_
            #print(self.classifier.best_score_)
            self.estimator = self.classifier.estimator
            search_result = self.estimator.fit(X_train, y_train)
            logger.debug("Model parameters = {}".format(self.classifier.get_params()))
            print("Model parameters = {}".format(self.classifier.get_params()))
            self.is_trained = True
            return
        
        print('Training classifier {}'.format(self.name))
        
        #if self.feature_ranking == True:
        #    self.select_feature(X_train, y_train)
            
        logger.info('Training classifier {}'.format(self.name))
        print('X_train.shape, y_train.shape', np.shape(X_train), np.shape(y_train))
        
#       main_tools.benchmark(self.classifier.fit, X_train, y_train) # fit the classifier with data
        logger.info('Trained classifier {}'.format(self.name))
        #self.training_error = self.classifier.score(X_train, y_train)

        print("Best: %f using %s" % (search_result.best_score_,search_result.best_params_))
        #grid_scores_：给出不同参数情况下的评价结果。best_params_：描述了已取得最佳结果的参数的组合
        #best_score_：成员提供优化过程期间观察到的最好的评分
        #具有键作为列标题和值作为列的dict，可以导入到DataFrame中。
        #注意，“params”键用于存储所有参数候选项的参数设置列表。
        #if self.param_grid is not None:
            #means = search_result.cv_results_['mean_test_score']
            #params = search_result.cv_results_['params']
            #for mean,param in zip(means,params):
                #print("%f  with:   %r" % (mean,param))
                #logger.debug("Grid search best score = {}".format(self.classifier.best_score_))
                #print("Grid search best score = {}".format(self.classifier.best_score_))
                #logger.debug("Grid search best estimator = {}".format(self.classifier.best_estimator_))
                #print("Grid search cv results = {}".format(self.classifier.cv_results_))
                #logger.debug("Grid search best estimator = {}".format(self.classifier.best_estimator_))
                #print("Grid search best estimator = {}".format(self.classifier.best_estimator_))
        #else:
            #logger.debug("Model parameters = {}".format(self.classifier.get_params()))
        

    def predict(self, X_test, y_test):
        
        if not self.is_trained:
            raise Exception('Model not trained, please run train()')

        self.score = self.estimator.score(X_test, y_test)
        if self.name == "Random Forest":
            self.oob_score = self.estimator.oob_score_
        if self.name in ("Random Forest", 'XGBoost', 'Decision Tree'):
            self.y_pred = [round(value) for value in self.estimator.predict(X_test)]
        else:
            self.y_pred = self.estimator.predict(X_test)
        # Call predict on the estimator (with the best found parameters if Grid search).
        # Round is there is we have probabilities (like with XGBoost)        
    
    def _metrics(self, cnf_matrix, metrics):
        
        tn, fp, fn, tp = cnf_matrix.ravel()
        self.tn, self.fp, self.fn, self.tp = tn, fp, fn, tp
        tpr = -1 if tp <= 0 else float(tp) / (tp + fn)
        metrics["TPR"] = tpr  # True Positive Rate recall
        
        tnr = -1 if tn <= 0 else float(tn) / (fp + tn)
        metrics["TNR"] = tnr  # True Negative Rate 
        
        fpr = -1 if tn <= 0 else float(fp) / (fp + tn)
        metrics["FPR"] = fpr  # False Positive Rate
        
        #fdr = -1 if tp <= 0 else float(fp) / (fp + tp)
        #self.metrics["FDR"] = fdr  # False Discovery Rate
        
        accuracy = -1 if tp <= 0 or tn <= 0 else float(tp + tn) / (tp + tn + fp + fn)
        metrics["Acc"] = accuracy
        
        error_rate = -1 if tp <= 0 or tn <= 0 else float(fp + fn) / (tp + fn + fp + tn)
        metrics["Err"] = error_rate
        
        precision = -1 if tp <= 0 else float(tp) / (tp + fp)
        metrics["Pre"] = precision  
        
        f_measure = -1 if precision <= 0 else float(2 * precision * tpr) / (precision + tpr)
        metrics["F-M"] = f_measure
        
        
    def compute_metrics(self, y_test):
        if self.y_pred is None:
            raise Exception('No prediction found, please run predict()')
       
        cnf_matrix = confusion_matrix(y_test, self.y_pred)                          
        self.cnf_matrix = cnf_matrix
        np.set_printoptions(precision = 2)
        class_names = [self.DIG2LABEL[i] for i in range(12)]
        # Plot non-normalized confusion matrix    
        FOLDER = self.model_path
        if not os.path.exists(FOLDER):
            os.mkdir(FOLDER)
        MODEL_PATH = FOLDER + '/model.ckpt'
        FIG_PATH = FOLDER + '/Confusion_Matrix.png'
        FIG_PATH_ROC = FOLDER + '/Plot_Roc_Dnn.png'
        FIG_PATH_N = FOLDER + '/Confusion_Matrix_Norm.png'
        FIG_PATH_VPN = FOLDER + '/VPN_Confusion_Matrix.png'
        FIG_PATH_N_VPN = FOLDER + '/VPN_Confusion_Matrix_Norm.png'        
        plt.figure()    
        plot_confusion_matrix(cm=cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
        plt.savefig(FIG_PATH)

        plt.figure()
        plot_confusion_matrix(cm=cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
        plt.savefig(FIG_PATH_N)
    
        #print(classification_report(y_t, y_p, target_names=class_names))
        report = classification_report(y_test, self.y_pred, target_names=class_names)
        self.metrics = classification_report_csv(report, FOLDER)            
    
        # vor VPN or non-VPN
        y_true = [self.DIG2LABEL[i] for i in y_test]
        y_preds = [self.DIG2LABEL[i] for i in self.y_pred]   
        labels = class_names
    
        y_vpn_true = np.where(y_test > 6, 'vpn', 'non-vpn')
        y_vpn_preds = np.where(np.array(self.y_pred) > 6, 'vpn', 'non-vpn')
        
        self.vpn_acc = len([v for i, v in enumerate(y_vpn_preds) if v == y_vpn_true[i]]) / len(y_vpn_true)
        
        #print('-----vpn_acc----:', vpn_acc)
        # Binarize the output
        y_vpn_true1 = label_binarize(y_vpn_true, classes=['non-vpn', 'vpn'])
        y_vpn_preds1 = label_binarize(y_vpn_preds, classes=['non-vpn', 'vpn'])
        n_classes = y_vpn_true1.shape[1]
        # Stream/non-stream ROC curve
        # utils.plot_ROC(y_vpn_true1, y_vpn_preds1, n_classes, ['non-vpn', 'vpn'],  FIG_PATH_ROC, micro=False, macro=False)

        cnf_matrix = confusion_matrix(y_vpn_true, y_vpn_preds, labels=['non-vpn', 'vpn'])
        roc_fpr, roc_tpr, thresholds = metrics.roc_curve(y_vpn_true1, y_vpn_preds1)            
        self.vpn_metrics["AUC"] = metrics.auc(roc_fpr, roc_tpr)         
        self._metrics(cnf_matrix, self.vpn_metrics)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=['non-vpn', 'vpn'], normalize=True, title='vpn non vpn Normalized confusion matrix')
        plt.savefig(FIG_PATH_VPN) 
    
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=['non-vpn', 'vpn'], title='non Normalized confusion matrix')
        plt.savefig(FIG_PATH_N_VPN)
    
        #print(classification_report(y_vpn_true1, y_vpn_preds1, target_names=['non-vpn', 'vpn']))    
        report = classification_report(y_vpn_true1, y_vpn_preds1, target_names=['non-vpn', 'vpn'])
        classification_report_csv(report, FOLDER, vpn=True)
        
        
    def get_printable_metrics(self):
        if len(self.metrics) == 0:
            raise Exception('No metrics found, please run compute_metrics()')

        """
        from prettytable import PrettyTable
        import operator

        headers = ['Model', 'Best score']
        headers += self.metrics.keys()

        table = PrettyTable(headers)
        content = [self.name, self.score]
        content += [round(float(m), 3) for m in self.metrics.values()]
        table.add_row(content)

        return table.get_string(sort_key=operator.itemgetter(2, 1), sortby="Best score", reversesort=True)
        """

        headers = ['Exec time', 'Model', 'Best score']
        headers += self.metrics.keys()
        print(self.metrics.values())
        values = time.strftime("%Y-%m-%d_%H-%M-%S") + "\t" + "\t".join([self.name, str(self.score)])
        for v in list(map(str, self.metrics.values())):
            values += ' '
            values += v
        return "\t".join(headers) + "\n" + values

    def visualization(self):
               
        if not os.path.exists(self.model_path):
            os.mkdir(out_path)
        feature_names = self.get_feature_name('ColSample.csv')
        
        if self.name in ('Decision Tree', 'Random Forest'):
            Estimators = self.estimator
            for index, model in enumerate(Estimators):
                dot_data = StringIO()
                filename = self.name + str(index) + '.pdf'
                export_graphviz(model, out_file=dot_data,
                                feature_names=feature_names,
                                class_names=list(self.LABEL2DIG.keys()),
                                filled=True, rounded=True,
                                special_characters=True)
                graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
                # 使用ipython的终端jupyter notebook显示。
                #Image(graph.create_png())
                graph.write_pdf(self.model_path + filename)
           
        if self.name == 'Decision Tree':
            dot_data = StringIO()
            export_graphviz(self.estimator, out_file = dot_data, feature_names = feature_names,
                                 class_names = list(self.LABEL2DIG.keys()), filled = True, rounded = True,
                                 special_characters = True)
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
            graph.write_pdf(self.model_path + "/Decision_Tree_Vision.pdf")            
       
        elif self.name == 'XGBoost':
            def create_feature_map(features):
                outfile = open('clf.fmap', 'w')
                for i, f in enumerate(features):
                    outfile.write('{0}\t{1}\tq\n'.format(i, f))
                    outfile.close()
            create_feature_map(feature_names)
            self.estimator.plot_tree(self.estimator, num_trees=0, fmap='clf.fmap')
            #fig = plt.gcf()
            fig.set_size_inches(120, 120)
            fig.savefig(self.model_path + '/XGB_Tree_Vision.png')        
        else:
            return

        
    @staticmethod
    def models_metric_summary(models):
        
        
        from cycler import cycler
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    
        # Define a list of markevery cases and color cases to plot
        colors = ['#1f77b4',
                  '#ff7f0e',
                  '#2ca02c',
                  '#d62728',
                  '#9467bd',
                  '#8c564b']
    
        # Configure rcParams axes.prop_cycle to simultaneously cycle cases and colors.
        #mpl.rcParams['axes.prop_cycle'] = cycler(markevery=cases, color=colors)
    
        # Create data points and offsets
        #x = np.linspace(0, 2 * np.pi)
        #offsets = np.linspace(0, 2 * np.pi, 11, endpoint=False)
        #yy = np.transpose([np.sin(x + phi) for phi in offsets])
    
        # Set the plot curve with markers and a title
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
        
        metricx_frame = pd.DataFrame(columns=models[0].metrics.columns)
        
        for i in range(len(models)):            
            metricx_frame.loc[models[i].name] = models[i].metrics.loc['avg']
            
            for j in range(len(models[i].columns - 1)):
                metric = models[i].columns[j + 1]
                ax.plot(models[i]['class'], models[i][metrics], marker='o', label=models[i].name)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
            plt.title('Metrics %s' % metric)    
            #plt.show()
            plt.savefig(models[i].model_path + '/metric/' )
  
        
        #from prettytable import PrettyTable
        #import operator
        
        #headers = ['Model', 'Best score']
        #headers += models[0].metrics.keys()
        """
        table = PrettyTable(headers)

        for model in models:
            if len(model.metrics) == 0:
                raise Exception('No metrics found for model "{}", please run compute_metrics()'.format(model.name))
            content = [model.name, model.score]
            content += [round(float(m), 3) for m in model.metrics.values()]
            table.add_row(content)

        return table.get_string(sort_key=operator.itemgetter(2, 1), sortby="Best score", reversesort=True)
        """

        #values = ""
        #for model in models:
            #if len(model.metrics) == 0:
                #raise Exception('No metrics found for model "{}", please run compute_metrics()'.format(model.name))

            #values += "\t".join([model.name, str(model.score)])
            #for v in list(map(str, model.metrics.values())):
                #values += ' '
                #values += v
        #return "\t".join(headers) + "\n" + values

    def save(self, filename):
        logger.info("Saving model to {}...".format(filename))
        pickle.dump(self.estimator, open(filename, "wb"))
        logger.info("Model saved to {}!".format(filename))

    def load(self, filename):
        logger.info("Loading model from {}...".format(filename))
        self.estimator = pickle.load(open(filename, "rb"))
        logger.info("Model loaded from {}!".format(filename))
    
    def get_feature_name(self, filename):
        features_names = pd.read_csv('../prepro/' + filename).columns.ravel()
        return features_names
