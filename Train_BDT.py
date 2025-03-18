import os
from util_bdt import *

class TrainBDT:
    def __init__(self, source_data_path: str, list_training_files: list,
                 output_path: str):
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
    
    def split_data(self, cols_input: list, cols_output: list, cols_output_regressor: list,
                   cols_output_classifier: list):
        self.cols_output_classifier = cols_output_classifier
        self.cols_output_regressor = cols_output_regressor
        self.splitted_data = train_valid_test(original_df=self.data, cols_input=cols_input, cols_output=cols_output,
                                cols_output_classifier=cols_output_classifier, cols_output_regressor=cols_output_regressor)

    def __regressor_GridSearch(self, splitted_data_regressor: dict, item_to_predict: str):
        param_grid = {
            'max_depth' : [5, 7, 10, 15, 20],
            'learning_rate': [0.6, 0.5, 0.4, 0.3, 0.2],
            'n_estimators': [100, 200],
            'min_child_weight' : [3, 5, 7],
            'subsample' : [1.0],
            'colsample_bytree' : [0.8],
        }
        best_params_regressor = gridSearch_Regressor(train_data_dict=splitted_data_regressor, param_grid=param_grid,
                                                     item_to_predict=item_to_predict)
        print(f'Best parameters for the regressor model : {best_params_regressor}')
        return best_params_regressor
    
    def RegressorModel(self, item_to_predict='max_deviation', saveModel=False):
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
            decay_factor=0.98,  # 5% decay
            decay_rounds=10     # every 50 rounds
        )

        xgb_regressor = xgb.train(params=best_params_regressor,
                          dtrain=dtrain,
                          evals=[(dtrain, 'train'), (dvalid, 'eval')],
                          early_stopping_rounds=50,
                          evals_result=eval_results,
                          callbacks=[lr_callback],
                          verbose_eval=True)
        if saveModel:
            xgb_regressor.save_model(f'{self.output_path}/{item_to_predict}_model.json')
        return xgb_regressor
    
    def test_regressor(self, xgb_regressor_model=None, item_to_predict=None):
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
        plt.hist(pred_df[item_to_predict], histtype='step', bins=100, label='max_deviation prediction')
        plt.hist(regressor_data['y_test'][item_to_predict], histtype='step', bins=100, label='max_deviation true')
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(f'{self.output_path}/{item_to_predict}_comparison_True_Prediction.png')
    

    def ClassifierModel(self, saveModel=False):
        classifier_data = self.splitted_data['classifier']
        # Convert to DMatrix for XGBoost
        dtrain = dataframe2DMatrix(classifier_data['X_train'], y=classifier_data['y_train'])
        dval = dataframe2DMatrix(classifier_data['X_val'], y=classifier_data['y_val'])
        # Set parameters
        params = {
            'learning_rate': 0.1, 'max_depth': 3,
            'objective': 'binary:logistic',  # for classification
            'eval_metric': 'logloss',
            'tree_method' : 'gpu_hist'
        }
        # Specify evaluation sets
        evals = [(dtrain, 'train'), (dval, 'validation')]

        # Train model
        model_classifier = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            # num_boost_round = 200,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=True
        )
        if saveModel:
            model_classifier.save_model(f'{self.output_path}/classifier_resp_model.json')
        return model_classifier
    
    def test_classifier(self, xgb_classifier_model=None, ):
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


def main():
    root_path = 'data/labelledData'
    output_path = 'OUTPUT'
    try:
        os.mkdir(output_path)
    except:
        pass
    xgb_obj = TrainBDT(source_data_path=root_path, list_training_files=os.listdir(root_path),
                       output_path=output_path)
    #
    cols_output_classifier = ['class_c1', 'class_c2', 'class_c3', 'class_c4']
    cols_input = ['A_0', 't_p', 'k3', 'k4', 'k5', 'k6']
    cols_output = cols_output_classifier + ['integral_R', 'max_deviation']
    cols_output_regressor = ['integral_R', 'max_deviation']
    xgb_obj.split_data(cols_input=cols_input, cols_output=cols_output, cols_output_classifier=cols_output_classifier,
                         cols_output_regressor=cols_output_regressor)
    #
    regressor_maxDev_model = xgb_obj.RegressorModel(item_to_predict='max_deviation', saveModel=True)
    xgb_obj.test_regressor(xgb_regressor_model=regressor_maxDev_model, item_to_predict='max_deviation')
    #
    regressor_int_model = xgb_obj.RegressorModel(item_to_predict='integral_R', saveModel=True)
    xgb_obj.test_regressor(xgb_regressor_model=regressor_int_model, item_to_predict='integral_R')
    #
    classifier_model = xgb_obj.ClassifierModel(saveModel=True)
    xgb_obj.test_classifier(xgb_classifier_model=classifier_model)

if __name__=='__main__':
    main()