import os
from util_bdt import *
from matplotlib.colors import LogNorm

class TrainBDT:
    def __init__(self, source_data_path: str, list_training_files: list,
                 output_path: str):
        """
            Initialize the TrainBDT class.
            
            Args:
                source_data_path (str): Path to training data files
                list_training_files (list): List of training file names
                output_path (str): Path where models and results will be saved
        """
        self.source_data_path = source_data_path
        self.list_training_files = list_training_files
        self.output_path = output_path

        self.data = self.read_data()

        self.cols_output_classifier=None
        self.cols_output_regressor=None
        self.splitted_data = dict() # output of self.split_data(...)

    def read_data(self):
        """
            This function reads the training files and concatenate them in a dataframe.
            It also one_hot_encode the classes for the classification.
        """
        data = pd.DataFrame()
        for i, f in enumerate(self.list_training_files):
            print(f, self.source_data_path)
            if i==0:
                data = pd.read_csv('/'.join([self.source_data_path, f]))
                columns = data.columns
                if columns[0] != '#Ch.#':
                    data.drop(columns=columns[0], inplace=True)
            else:
                tmpdata = pd.read_csv('/'.join([self.source_data_path, f]))
                columns = tmpdata.columns
                if columns[0] != '#Ch.#':
                    tmpdata.drop(columns=columns[0], inplace=True)
                data = pd.concat([data, tmpdata], axis=0)
        return one_hot_encode_sklearn(data=data, column_name='class')
        # return data
    
    def split_data(self, cols_input: list, cols_output: list, cols_output_regressor: list,
                   cols_output_classifier: list):
        """
            Split the data into training, validation and test sets for both classifier and regressor.
            
            Args:
                cols_input (list): Feature columns used for training
                cols_output (list): All output columns (classifier + regressor)
                cols_output_regressor (list): Columns for regression tasks
                cols_output_classifier (list): Columns for classification tasks
        """
        self.cols_output_classifier = cols_output_classifier
        self.cols_output_regressor = cols_output_regressor
        self.splitted_data = train_valid_test(original_df=self.data, cols_input=cols_input, cols_output=cols_output,
                                cols_output_classifier=cols_output_classifier, cols_output_regressor=cols_output_regressor)
        ###
        # ### print number of each class 
        print(f'training : {len(self.splitted_data['classifier']['y_train'])}')
        print(f'training class_c1 : {len(self.splitted_data['classifier']['y_train'][self.splitted_data['classifier']['y_train']['class_c1']==1.0])}')
        print(f'training class_c2 : {len(self.splitted_data['classifier']['y_train'][self.splitted_data['classifier']['y_train']['class_c2']==1.0])}')
        print(f'training class_c3 : {len(self.splitted_data['classifier']['y_train'][self.splitted_data['classifier']['y_train']['class_c3']==1.0])}')
        print(f'training class_c4 : {len(self.splitted_data['classifier']['y_train'][self.splitted_data['classifier']['y_train']['class_c4']==1.0])}')
        #
        print(f'validation class_c1 : {len(self.splitted_data['classifier']['y_val'][self.splitted_data['classifier']['y_val']['class_c1']==1.0])}')
        print(f'validation class_c2 : {len(self.splitted_data['classifier']['y_val'][self.splitted_data['classifier']['y_val']['class_c2']==1.0])}')
        print(f'validation class_c3 : {len(self.splitted_data['classifier']['y_val'][self.splitted_data['classifier']['y_val']['class_c3']==1.0])}')
        print(f'validation class_c4 : {len(self.splitted_data['classifier']['y_val'][self.splitted_data['classifier']['y_val']['class_c4']==1.0])}')
        #
        print(f'testing class_c1 : {len(self.splitted_data['classifier']['y_test'][self.splitted_data['classifier']['y_test']['class_c1']==1.0])}')
        print(f'testing class_c2 : {len(self.splitted_data['classifier']['y_test'][self.splitted_data['classifier']['y_test']['class_c2']==1.0])}')
        print(f'testing class_c3 : {len(self.splitted_data['classifier']['y_test'][self.splitted_data['classifier']['y_test']['class_c3']==1.0])}')
        print(f'testing class_c4 : {len(self.splitted_data['classifier']['y_test'][self.splitted_data['classifier']['y_test']['class_c4']==1.0])}')
        input('Press enter to continue....')

    def __regressor_GridSearch(self, splitted_data_regressor: dict, item_to_predict: str, regressor=True):
        """
            Perform grid search to find optimal hyperparameters for the regressor model.
            
            Args:
                splitted_data_regressor (dict): Dictionary containing split data for regression
                item_to_predict (str): Target variable name for regression
                
            Returns:
                dict: Best parameters found during grid search
        """
        param_grid = {
            # 'max_depth' : [5, 7, 10, 15, 20],
            # 'learning_rate': [0.6, 0.5, 0.4, 0.3, 0.2],
            # 'n_estimators': [100, 200],
            # 'min_child_weight' : [3, 5, 7],
            # 'subsample' : [1.0],
            # 'colsample_bytree' : [0.8],
            'max_depth': [15],#, 20],
            'learning_rate': [0.4, 0.3],
            # 'n_estimators': [50, 100, 150],
            'min_child_weight' : [15],#, 20],
            'subsample': [1.0],
            'colsample_bytree': [1.0]
        }
        best_params_regressor = gridSearch_Regressor(train_data_dict=splitted_data_regressor, param_grid=param_grid,
                                                     item_to_predict=item_to_predict, regressor=regressor)
        print(f'Best parameters for the model : {best_params_regressor}')
        return best_params_regressor
    
    def RegressorModel(self, item_to_predict='max_deviation', saveModel=False):
        """
            Train an XGBoost regressor model with learning rate decay.
            
            Args:
                item_to_predict (str): Target variable for regression
                saveModel (bool): Whether to save the trained model
                
            Returns:
                xgb.Booster: Trained XGBoost regressor model
        """
        regressor_data = self.splitted_data['regressor']
        best_params_regressor = self.__regressor_GridSearch(splitted_data_regressor=regressor_data, item_to_predict=item_to_predict)

        dtrain = dataframe2DMatrix(X=regressor_data['X_train'],y=regressor_data['y_train'][item_to_predict])
        dvalid = dataframe2DMatrix(X=regressor_data['X_val'], y=regressor_data['y_val'][item_to_predict])
        eval_results = {}

        # initial learning rate
        initial_lr = best_params_regressor.get('learning_rate', 0.3)
        # Create proper callback instance
        lr_callback = LearningRateDecay(
            initial_lr=initial_lr,
            decay_factor=0.75,  # 5% decay
            decay_rounds=20     # every 50 rounds
        )

        xgb_regressor = xgb.train(params=best_params_regressor,
                          dtrain=dtrain,
                          evals=[(dtrain, 'train'), (dvalid, 'eval')],
                          early_stopping_rounds=100,
                          evals_result=eval_results,
                          callbacks=[lr_callback],
                          verbose_eval=True)
        if saveModel:
            xgb_regressor.save_model(f'{self.output_path}/{item_to_predict}_model.json')
        return xgb_regressor
    
    def test_regressor(self, xgb_regressor_model=None, item_to_predict=None):
        """
            Evaluate the regressor model on test data and generate performance metrics.
            
            Args:
                xgb_regressor_model: Trained XGBoost regressor model
                item_to_predict (str): Target variable being predicted
                
            Outputs:
                - Saves predictions to CSV
                - Generates comparison plots
                - Prints MSE score
        """
        if xgb_regressor_model==None:
            return None
        regressor_data = self.splitted_data['regressor']
        dtest = dataframe2DMatrix(X=regressor_data['X_test'])
        predictions = xgb_regressor_model.predict(dtest)
        # MSE
        mse = mean_absolute_error(regressor_data['y_test'][item_to_predict], pd.DataFrame(predictions, columns=[item_to_predict]))
        print(f'MSE = {mse}')
        #
        # predictions in a dataframe
        pred_df = pd.DataFrame(predictions, columns=[item_to_predict], index=regressor_data['y_test'][item_to_predict].index)
        pred_df.to_csv(f'{self.output_path}/{item_to_predict}_predicted.csv', index=True)
        #
        # Comparison between true values and predictions
        plt.figure()
        plt.hist(pred_df[item_to_predict], histtype='step', bins=100, label=f'{item_to_predict} prediction')
        plt.hist(regressor_data['y_test'][item_to_predict], histtype='step', bins=100, label=f'{item_to_predict} true')
        plt.xlabel(item_to_predict)
        plt.ylabel('#')
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(f'{self.output_path}/{item_to_predict}_comparison_True_Prediction.png')
    

    def ClassifierModel(self, saveModel=False):
        """
            Train an XGBoost classifier model for multi-class classification.
            
            Args:
                saveModel (bool): Whether to save the trained model
                
            Returns:
                xgb.Booster: Trained XGBoost classifier model
        """
        classifier_data = self.splitted_data['classifier']
        # Convert to DMatrix for XGBoost
        dtrain = dataframe2DMatrix(classifier_data['X_train'], y=classifier_data['y_train'])
        dval = dataframe2DMatrix(classifier_data['X_val'], y=classifier_data['y_val'])
        # Set parameters
        best_params_classifier = self.__regressor_GridSearch(splitted_data_regressor=classifier_data, item_to_predict='', regressor=False)
        #
        # Specify evaluation sets
        evals = [(dtrain, 'train'), (dval, 'validation')]

        # initial learning rate
        initial_lr = best_params_classifier.get('learning_rate', 0.3)
        # Create proper callback instance
        lr_callback = LearningRateDecay(
            initial_lr=initial_lr,
            decay_factor=0.75,  
            decay_rounds=100    
        )

        # Train model
        model_classifier = xgb.train(
            params=best_params_classifier,
            dtrain=dtrain,
            # num_boost_round=1000,
            evals=evals,
            callbacks=[lr_callback],
            early_stopping_rounds=100,
            verbose_eval=True,
        )
        if saveModel:
            model_classifier.save_model(f'{self.output_path}/classifier_resp_model.json')
        return model_classifier
    
    def test_classifier(self, xgb_classifier_model=None):
        """
            Evaluate the classifier model on test data and generate performance metrics.
            
            Args:
                xgb_classifier_model: Trained XGBoost classifier model
                
            Outputs:
                - Saves predictions to CSV
                - Generates confusion matrix plot
                - Prints classification accuracy
        """
        if xgb_classifier_model is None:
            return None
        classifier_data = self.splitted_data['classifier']
        dtest = dataframe2DMatrix(classifier_data['X_test'])
        # Make predictions
        preds = xgb_classifier_model.predict(dtest)
        # to dataframe
        columns = self.cols_output_classifier
        predClass_df = pd.DataFrame(preds, index=classifier_data['X_test'].index, columns=columns)
        # create a new column : predicted_class
        predClass_df['predicted_class'] = predClass_df.idxmax(axis=1)
        #
        # concatenate the truth and the predictions
        test_df = pd.concat([classifier_data['X_test'], pd.DataFrame(classifier_data['y_test'].idxmax(axis=1), columns=['trueClass']), predClass_df['predicted_class']], axis=1)
        # Overall accuracy of the prediction. This calculation doesn't take into account the imbalance in the dataset
        Accuracy = (np.sum(test_df['trueClass']==test_df['predicted_class'])/len(test_df['trueClass']))*100
        print(f'Accuracy of the prediction = {np.round(Accuracy,5)}%')
        #
        # Confusion matrix of the classified data
        # Create confusion matrix
        cm = confusion_matrix(y_true=test_df['trueClass'], y_pred=test_df['predicted_class'])

        # Create a figure and axis
        plt.figure(figsize=(10, 8))

        # Create heatmap
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='crest',
                    xticklabels=self.cols_output_classifier,
                    yticklabels=self.cols_output_classifier)

        # Add labels
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix')

        # Show plot
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{self.output_path}/ConfusionMatrix_True_vs_Prediction.png')

        # save the classified test data to csv file
        test_df.to_csv(f'{self.output_path}/classification_test_data.csv', index=True)

    def outputRegressor2Classifier(self, listfilenames=None):
        # read the data that were predicted by the regressor models
        data = pd.DataFrame()
        for i,f in enumerate(listfilenames):
            tmpdata = pd.read_csv('/'.join([self.output_path, f]))#.reset_index().drop(columns='index', inplace=False)
            columns = tmpdata.columns
            tmpdata.columns = ['index', columns[1]]
            tmpdata.index = tmpdata['index']
            tmpdata.drop(columns='index', inplace=True)
            tmpdata.index.name = None
            if i==0:
                data = tmpdata.copy()
                # data.index = data['index']
                # data.drop(columns='index', inplace=True)
                # data.index.name = None
            else:
                # data = pd.merge(data, tmpdata, on='index', how='inner')
                data = pd.concat([data, tmpdata], axis=1, join='outer')
                # print(data)
                # sys.exit()
        # data = data.reset_index().drop(columns='level_0', inplace=False)
        columns = []
        for c in data.columns:
            if c!= 'index':
                columns.append(f'true_{c}')
            else:
                columns.append(c)
        # Truth information
        classifier_data = self.splitted_data['classifier']
        c = classifier_data['X_test'].copy()#.reset_index()
        c.columns = columns
        # input_classifier_df  = pd.merge(data, c, on='index', how='inner')
        input_classifier_df = pd.concat([data, c], axis=1, join='outer')
        # input_classifier_df = pd.merge(input_classifier_df,  classifier_data['y_test'].reset_index(), on='index', how='inner')
        input_classifier_df = pd.concat([input_classifier_df, classifier_data['y_test']], axis=1, join='outer')
        # input_classifier_df = input_classifier_df.set_index('index', drop=True)
        # input_classifier_df.index.name = None
        #
        # 2D correlation plots between truth and predicted values from the regression models
        output_plot_path = f'{self.output_path}/plots'
        try:
            os.mkdir(output_plot_path)
        except:
            pass
        columns = [c for c in input_classifier_df.columns if 'class' in c]
        for c in columns:
            class_df = input_classifier_df[input_classifier_df[c]==1.0]
            fig, ax = plt.subplots(1,2, figsize=(8*2, 8))
            h1 = ax[0].hist2d(class_df['integral_R'], class_df['true_integral_R'], 
                    bins=30, norm=LogNorm(),cmap='viridis')
            cbar1 = plt.colorbar(h1[3])
            cbar1.set_label('Counts')
            ax[0].set_xlabel('predicted integral_R')
            ax[0].set_ylabel('true integral_R')
            ax[0].set_title(c)
            
            h2 = ax[1].hist2d(class_df['max_deviation'], class_df['true_max_deviation'], 
                    bins=30, norm=LogNorm(),cmap='viridis')
            cbar2 = plt.colorbar(h2[3])
            cbar2.set_label('Counts')
            ax[1].set_xlabel('predicted max_deviation')
            ax[1].set_ylabel('true max_deviation')
            ax[1].set_title(c)

            plt.tight_layout()
            plt.savefig(f'{output_plot_path}/{c}_2dCorr.png')
            plt.close()
        #
        # 1D plots of the difference between prediction and truth
        for c in columns:
            class_df = input_classifier_df[input_classifier_df[c]==1.0]
            fig, ax = plt.subplots(1,2,figsize=(8*2,8))
            ax[0].hist(class_df['integral_R'] - class_df['true_integral_R'], histtype='step')
            ax[0].set_xlabel('predicted - true values')
            ax[0].set_ylabel('#')
            ax[0].set_title(f'{c} : integral_R')
            ax[0].grid(True)
        
            ax[1].hist(class_df['max_deviation'] - class_df['true_max_deviation'], histtype='step')
            ax[1].set_xlabel('predicted - true values')
            ax[1].set_ylabel('#')
            ax[1].set_title(f'{c} : max_deviation')
            ax[1].grid(True)
            plt.tight_layout()
            plt.savefig(f'{output_plot_path}/{c}_1dDiff.png')
        ##
        ## Attempt to classify
        dtest_fromPrediction = dataframe2DMatrix(input_classifier_df[['integral_R', 'max_deviation']])
        classifier_model = xgb.Booster()
        classifier_model.load_model(f'{self.output_path}/classifier_resp_model.json')
        pred_classes = classifier_model.predict(dtest_fromPrediction)
        #
        columns = [cl for cl in input_classifier_df.columns if 'class' in cl]
        # predClass_df = pd.DataFrame(pred_classes, index=classifier_data['X_test'].index, columns=columns)
        predClass_df = pd.DataFrame(pred_classes, index=input_classifier_df.index, columns=columns)
        predClass_df['predicted_class'] = predClass_df.idxmax(axis=1)
        #
        X_cols = [cl for cl in input_classifier_df.columns if 'class' not in cl]
        test_df = pd.concat([input_classifier_df[X_cols], pd.DataFrame(input_classifier_df[columns].idxmax(axis=1), columns=['trueClass']), predClass_df['predicted_class']], axis=1)
        #
        test_df.to_csv(f'{output_plot_path}/classification_of_predictedMaxDev_IntR.csv', index=True)
        # Overall accuracy of the classification
        Accuracy = (np.sum(test_df['trueClass']==test_df['predicted_class'])/len(test_df['trueClass']))*100
        print(f'Accuracy of the prediction = {np.round(Accuracy,5)}%')
        #
        # Create confusion matrix
        cm = confusion_matrix(y_true=test_df['trueClass'], y_pred=test_df['predicted_class'])

        # Create a figure and axis
        plt.figure(figsize=(10, 8))

        # Create heatmap
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='crest',
                    # xticklabels=['Overshoot', 'Undershoot', 'Ideal', 'Singularity', 'Abnormal'],  # If your classes are 0-4
                    # yticklabels=['Overshoot', 'Undershoot', 'Ideal', 'Singularity', 'Abnormal'])  # If your classes are 0-4
                    xticklabels=columns,
                    yticklabels=columns)

        # Add labels
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix')

        # Show plot
        plt.tight_layout()
        plt.savefig(f'{output_plot_path}/confusionMatrix_onOutputRegressor.png')

def TestClassifier(path_to_model, path_to_data, datakey, colsInput=['integral_R', 'max_deviation'], trueInput=True, output_path=None):
    """
        datakey: is a key in the filename unique for the dataset.
        trueInput should be set to False when using an unlabelled dataset.
    """
    output_plot_path = f'{output_path}/plots'
    try:
        os.mkdir(output_plot_path)
    except:
        pass
    # read data
    data = pd.DataFrame()
    listfiles = [f for f in os.listdir(path_to_data) if (datakey in f) and ('.csv' in f)]
    if trueInput:
        for i, f in enumerate(listfiles):
            if i==0:
                data = pd.read_csv(f'{path_to_data}/{f}')
            else:
                data = pd.concat([data, pd.read_csv(f'{path_to_data}/{f}')], axis=0)
        data = one_hot_encode_sklearn(data=data, column_name='class')
        data.drop(columns='Unnamed: 0', inplace=True)
    else:
        for i, f in enumerate(listfiles):
            if i==0:
                data = pd.read_csv(f'{path_to_data}/{f}')
            else:
                data = pd.merge(data, pd.read_csv(f'{path_to_data}/{f}'), how='inner', on=['#Ch.#', 'class'])
    data = one_hot_encode_sklearn(data=data, column_name='class')
    # tmp = data[colsInput].copy()
    # tmp = tmp / np.abs(tmp).max()
    # load model
    dtest_fromPrediction = dataframe2DMatrix(data[colsInput])
    # dtest_fromPrediction = dataframe2DMatrix(tmp)
    classifier_model = xgb.Booster()
    classifier_model.load_model(path_to_model)
    pred_classes = classifier_model.predict(dtest_fromPrediction)
    #
    columns = [cl for cl in data.columns if 'class' in cl]
    predClass_df = pd.DataFrame(pred_classes, index=data.index, columns=columns)
    predClass_df['predicted_class'] = predClass_df.idxmax(axis=1)
    #
    #
    X_cols = [cl for cl in data.columns if 'class' not in cl]
    test_df = pd.concat([data[X_cols], pd.DataFrame(data[columns].idxmax(axis=1), columns=['trueClass']), predClass_df['predicted_class']], axis=1, join='inner')
    #
    test_df.to_csv(f'{output_plot_path}/classification_of_predictedMaxDev_IntR.csv', index=False)
    # Overall accuracy of the classification
    Accuracy = (np.sum(test_df['trueClass']==test_df['predicted_class'])/len(test_df['trueClass']))*100
    print(f'Accuracy of the prediction = {np.round(Accuracy,5)}%')
    #
    # Create confusion matrix
    cm = confusion_matrix(y_true=test_df['trueClass'], y_pred=test_df['predicted_class'])

    # Create a figure and axis
    plt.figure(figsize=(10, 8))

    # Create heatmap
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='crest',
                xticklabels=columns,
                yticklabels=columns)

    # Add labels
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')

    # Show plot
    plt.tight_layout()
    plt.savefig(f'{output_plot_path}/confusionMatrix_onOutputRegressor.png')
    plt.close()

def TestRegressor(path_to_model, path_to_data, datakey, colsInput=['A_0', 'k3'], isValidation=True, output_path=None, item_to_predict='integral_R'):
    """
        datakey: is a key in the filename unique for the dataset.
        isValidation should be set to False when using an unlabelled dataset.
    """
    output_plot_path = f'{output_path}/plots'
    try:
        os.mkdir(output_plot_path)
    except:
        pass
    # read data
    data = pd.DataFrame()
    listfiles = [f for f in os.listdir(path_to_data) if (datakey in f) and ('.csv' in f)]    
    for i, f in enumerate(listfiles):
        if i==0:
            data = pd.read_csv(f'{path_to_data}/{f}')
        else:
            data = pd.concat([data, pd.read_csv(f'{path_to_data}/{f}')], axis=0)
    data_copy = data.copy()
    data = one_hot_encode_sklearn(data=data, column_name='class')
    dtest = dataframe2DMatrix(data[colsInput])
    #
    regressor_model = xgb.Booster()
    regressor_model.load_model(path_to_model)
    predictions = regressor_model.predict(dtest)
    # MSE
    mse = mean_absolute_error(data[item_to_predict], pd.DataFrame(predictions, columns=[item_to_predict]))
    print(f'MSE = {mse}')
    #
    # predictions in a dataframe
    pred_df = pd.DataFrame(predictions, columns=[item_to_predict], index=data[item_to_predict].index)
    columns = [cl for cl in data.columns if 'class' in cl]

    pred_df = pd.concat([pred_df, data['#Ch.#'], data_copy['class']], axis=1, join='outer')
    pred_df.to_csv(f'{output_path}/{item_to_predict}_predicted.csv', index=False)
    #
    # Comparison between true values and predictions
    plt.figure(figsize=(12, 12))
    plt.hist(pred_df[item_to_predict], histtype='step', bins=100, label=f'{item_to_predict} prediction', linewidth=3)
    plt.hist(data[item_to_predict], histtype='step', bins=100, label=f'{item_to_predict} true', linewidth=3)
    plt.xlabel(item_to_predict)
    plt.ylabel('#')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f'{output_path}/plots/{item_to_predict}_comparison_True_Prediction.png')
    plt.close()

def main():
    """
        Main function to demonstrate the usage of TrainBDT class.
        Trains and evaluates both regressor and classifier models.
    """
    plt.rcParams.update({
            'font.size': 18,
            'axes.titlesize': 15,
            'axes.labelsize': 15,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 18,
            'figure.titlesize': 20
        })
    
    # root_path = 'data/labelledData/labelledData/generatedSamples'
    root_path = 'data/labelledData/labelledData'
    # root_path = 'data/labelledData_after_March22_2025'
    output_path = 'OUTPUT/synthetic'
    try:
        os.mkdir(output_path)
    except:
        pass
    list_files = [f for f in os.listdir(root_path) if ('.csv' in f) and ('kde' not in f)]
    # xgb_obj = TrainBDT(source_data_path=root_path, list_training_files=[f for f in os.listdir(root_path) if ('.csv' in f) and ('30413' in f)],
    #                    output_path=output_path)
    xgb_obj = TrainBDT(source_data_path=root_path, list_training_files=list_files,
                       output_path=output_path)
    #
    cols_output_classifier = ['class_c1', 'class_c2', 'class_c3', 'class_c4']
    # cols_output_classifier = ['class_c1', 'class_c3', 'class_c4']
    
    cols_input = ['A_0', 't_p', 'k3', 'k4', 'k5', 'k6']
    # # cols_input = ['t_p', 'k3', 'k4', 'k5', 'k6']
    cols_output = cols_output_classifier + ['integral_R', 'max_deviation']
    cols_output_regressor = ['integral_R', 'max_deviation']
    xgb_obj.split_data(cols_input=cols_input, cols_output=cols_output, cols_output_classifier=cols_output_classifier,
                         cols_output_regressor=cols_output_regressor)
    # #
    regressor_maxDev_model = xgb_obj.RegressorModel(item_to_predict='max_deviation', saveModel=True)
    xgb_obj.test_regressor(xgb_regressor_model=regressor_maxDev_model, item_to_predict='max_deviation')
    #
    regressor_int_model = xgb_obj.RegressorModel(item_to_predict='integral_R', saveModel=True)
    xgb_obj.test_regressor(xgb_regressor_model=regressor_int_model, item_to_predict='integral_R')
    
    classifier_model = xgb_obj.ClassifierModel(saveModel=True)
    xgb_obj.test_classifier(xgb_classifier_model=classifier_model)
    xgb_obj.outputRegressor2Classifier(listfilenames=[f for f in os.listdir(output_path) if 'predicted.csv' in f])
    ##
    ## TESTING AFTER THE MODELS ARE TRAINED
    TestRegressor(path_to_model='OUTPUT/synthetic/integral_R_model.json', path_to_data='data/labelledData', datakey='fit_results', colsInput=cols_input, isValidation=True, output_path='OUTPUT/synthetic/TestOnData', item_to_predict='integral_R')
    TestRegressor(path_to_model='OUTPUT/synthetic/max_deviation_model.json', path_to_data='data/labelledData', datakey='fit_results', colsInput=cols_input, isValidation=True, output_path='OUTPUT/synthetic/TestOnData', item_to_predict='max_deviation')
    TestClassifier(path_to_model='OUTPUT/synthetic/classifier_resp_model.json', path_to_data='OUTPUT/synthetic/TestOnData', datakey='predicted', colsInput=['integral_R', 'max_deviation'], trueInput=False,
                   output_path='OUTPUT/synthetic/TestOnData')
if __name__=='__main__':
    main()