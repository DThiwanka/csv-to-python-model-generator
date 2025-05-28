
import { GoogleGenAI, GenerateContentResponse } from '@google/genai';
import { 
    CSVRow, ModelType, Algorithm, HyperparameterTuningMethod, 
    TrainTestSplitRatio, StratifyOption, ModelPersistenceOption,
    EvaluationMetric, EvaluationVisualization, ClassificationMetric, RegressionMetric,
    ClassificationVisualization, RegressionVisualization,
    // Advanced Preprocessing Types
    OutlierMethod, OutlierHandling, ColumnConversionConfig,
    DatetimeExtractionConfig, TextColumnConfig, TargetTransformationMethod
} from '../types';

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  console.error("API_KEY environment variable is not set. Gemini API calls will fail.");
}

const ai = API_KEY ? new GoogleGenAI({ apiKey: API_KEY }) : null;
const TEXT_MODEL_NAME = 'gemini-2.5-flash-preview-04-17';


export const suggestTargetVariable = async (headers: string[]): Promise<string> => {
  if (!ai) throw new Error("Gemini API Client not initialized. API_KEY might be missing.");
  const prompt = `I have a CSV file with the following column headers: ${JSON.stringify(headers)}.
I want to build a machine learning model to make predictions.
Which of these columns would typically be the best candidate for the target variable (the variable to be predicted)?
Please respond with ONLY the name of the column you recommend. For example, if 'Sales' is a good candidate, just respond 'Sales'.`;

  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
      model: TEXT_MODEL_NAME,
      contents: prompt,
      config: { temperature: 0.3 } 
    });
    
    const suggestedTarget = response.text.trim();
    if (headers.includes(suggestedTarget)) {
      return suggestedTarget;
    }
    console.warn(`Gemini suggestion '${suggestedTarget}' not in headers or empty. Falling back.`);
    return headers[headers.length - 1] || headers[0] || "unknown_target";
  } catch (error) {
    console.error("Error suggesting target variable:", error);
    throw new Error(`Failed to get target variable suggestion from AI: ${error instanceof Error ? error.message : String(error)}`);
  }
};

export const generatePythonModelCode = async (
  allCsvHeaders: string[], // All original headers
  targetVariable: string,
  sampleData: CSVRow[],
  // Advanced Preprocessing
  outlierMethod: OutlierMethod,
  outlierHandling: OutlierHandling,
  columnConversions: ColumnConversionConfig[],
  datetimeExtractions: DatetimeExtractionConfig[],
  textColumnConfigs: TextColumnConfig[],
  targetTransformation: TargetTransformationMethod,
  // Model Config
  modelType: ModelType,
  algorithm: Algorithm,
  tuningMethod: HyperparameterTuningMethod,
  splitRatio: TrainTestSplitRatio,
  stratifyOption: StratifyOption,
  modelPersistenceOption: ModelPersistenceOption,
  selectedMetrics: EvaluationMetric[],
  selectedVisualizations: EvaluationVisualization[]
): Promise<string> => {
  if (!ai) throw new Error("Gemini API Client not initialized. API_KEY might be missing.");

  const featureHeaders = allCsvHeaders.filter(h => h !== targetVariable);
  const csvHeadersString = allCsvHeaders.join(', ');
  
  let csvSampleString = allCsvHeaders.join(',') + '\\n';
  csvSampleString += sampleData.map(row => 
    allCsvHeaders.map(header => {
      const val = row[header];
      if (val === null || val === undefined) return '';
      let strVal = String(val);
      if (strVal.includes(',')) strVal = `"${strVal}"`;
      return strVal;
    }).join(',')
  ).join('\\n');

  const testSizeMapping: Record<TrainTestSplitRatio, number> = {
    '80/20': 0.2,
    '75/25': 0.25,
    '70/30': 0.3,
  };
  const testSize = testSizeMapping[splitRatio];
  const stratifyInstruction = (modelType === 'classification' && stratifyOption === 'yes') ? 'stratify=y' : 'stratify=None';
  const trainTestSplitInstruction = `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${testSize}, random_state=42, ${stratifyInstruction})`;

  const modelPersistenceInstruction = modelPersistenceOption === 'yes'
    ? `After training, evaluation, and potential hyperparameter tuning, save the final trained model to a file named 'trained_model.joblib' using the 'joblib' library. Import 'joblib' at the start. Include a comment on how to load it back.`
    : `Do NOT include code to save the model to a file.`;
  
  const algoNameMapping: Record<Algorithm, string> = {
    'logistic_regression': 'LogisticRegression',
    'random_forest_classifier': 'RandomForestClassifier',
    'svm_classifier': 'SVC',
    'gradient_boosting_classifier': 'GradientBoostingClassifier',
    'knn_classifier': 'KNeighborsClassifier',
    'xgboost_classifier': 'XGBClassifier',
    'linear_regression': 'LinearRegression',
    'random_forest_regressor': 'RandomForestRegressor',
    'svm_regressor': 'SVR',
    'gradient_boosting_regressor': 'GradientBoostingRegressor',
    'knn_regressor': 'KNeighborsRegressor',
    'xgboost_regressor': 'XGBRegressor',
  };
  const specificAlgorithmName = algoNameMapping[algorithm] || algorithm;

  // ADVANCED PREPROCESSING PROMPT SECTIONS
  let advancedPreprocessingInstructions = "\n# --- Advanced Preprocessing Steps (if any) ---";
  
  if (targetTransformation === 'log_transform') {
    advancedPreprocessingInstructions += `
# Apply Log Transformation to target variable '${targetVariable}'
# This is typically for skewed numeric targets in regression.
# Ensure to apply inverse transformation (np.expm1) to predictions if this is used.
if pd.api.types.is_numeric_dtype(df['${targetVariable}']):
    df['${targetVariable}'] = np.log1p(df['${targetVariable}'])
    print(f"Applied log1p transformation to target variable '${targetVariable}'.")
else:
    print(f"Warning: Log transformation selected for non-numeric target '${targetVariable}'. Skipping.")
`;
  }

  advancedPreprocessingInstructions += "\n# Initial definition of X and y before loop-based modifications for outlier handling\ny = df['${targetVariable}']\nX = df.drop(columns=['${targetVariable}'])\n"

  if (outlierMethod !== 'none' && outlierHandling === 'remove') {
    advancedPreprocessingInstructions += `
# Outlier Detection and Removal (${outlierMethod}) for numeric features in X
# This operates on X and y defined just above.
numeric_features_for_outliers = X.select_dtypes(include=np.number).columns
for col in numeric_features_for_outliers:
    # Ensure column still exists in X (it might have been modified or dropped by other steps if not careful with order)
    if col not in X.columns: continue
`;
    if (outlierMethod === 'z_score') {
      advancedPreprocessingInstructions += `
    # Calculate Z-scores only for non-NaN values
    col_mean = X[col].mean()
    col_std = X[col].std()
    if col_std == 0 : continue # Skip if standard deviation is zero
    z_scores = np.abs((X[col].dropna() - col_mean) / col_std)
    
    # Create a boolean mask for outliers, default to False (not outlier)
    outlier_mask_col = pd.Series(False, index=X.index)
    # Update mask for non-NaN values that are outliers
    outlier_mask_col[z_scores[z_scores >= 3].index] = True

    is_not_outlier = ~outlier_mask_col
`;
    } else if (outlierMethod === 'iqr') {
      advancedPreprocessingInstructions += `
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Create mask for non-outliers, considering NaNs as non-outliers for this step
    is_not_outlier = ((X[col] >= lower_bound) & (X[col] <= upper_bound)) | X[col].isna()
`;
    }
    advancedPreprocessingInstructions += `
    if 'is_not_outlier' in locals() and not is_not_outlier.all(): # check if any outlier was detected
        original_len = len(X)
        X = X[is_not_outlier]
        y = y[is_not_outlier.reindex(y.index, fill_value=True)] # Align y's index with X
        if len(X) < original_len:
            print(f"Removed {original_len - len(X)} outliers affecting column '{col}' based on ${outlierMethod}.")
`;
    advancedPreprocessingInstructions += `
# Reset index for X and y after all outlier loops are done
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
`;
  } else if (outlierMethod !== 'none') {
    advancedPreprocessingInstructions += `\n# Outlier detection method '${outlierMethod}' was selected, but handling is 'none'. Code for detection could be added here if desired for reporting.\n`;
  }


  columnConversions.forEach(conv => {
    advancedPreprocessingInstructions += `
# Forcing column '${conv.columnName}' to type '${conv.convertTo}'
# This operates on the main DataFrame 'df'. X and y will be redefined after these ops.
if '${conv.columnName}' in df.columns:
    try:
        if '${conv.convertTo}' == 'numeric':
            df['${conv.columnName}'] = pd.to_numeric(df['${conv.columnName}'], errors='coerce') 
        elif '${conv.convertTo}' == 'datetime':
            df['${conv.columnName}'] = pd.to_datetime(df['${conv.columnName}'], errors='coerce') 
        elif '${conv.convertTo}' == 'categorical' or '${conv.convertTo}' == 'text':
            df['${conv.columnName}'] = df['${conv.columnName}'].astype(str)
        print(f"Attempted to convert column '${conv.columnName}' to ${conv.convertTo}.")
    except Exception as e:
        print(f"Error converting column '${conv.columnName}' to ${conv.convertTo}: {e}")
else:
    print(f"Warning: Column '${conv.columnName}' not found in DataFrame for type conversion.")
`;
  });
  
  // Re-define X and y after potential type conversions on df and outlier removal from X,y copies
  advancedPreprocessingInstructions += `
# Re-define X and y using the potentially modified 'df' for type conversions,
# and use the X, y that might have had outliers removed.
# If outlier removal happened, X and y are already copies. If not, make copies from df.
if '${outlierMethod}' == 'none' or '${outlierHandling}' == 'none': # if no outlier removal, X and y need to be fresh from potentially type-converted df
    y = df['${targetVariable}'].copy()
    X = df.drop(columns=['${targetVariable}']).copy()
# else, X and y from outlier removal are used. They should reflect changes in df if outlier removal didn't execute.
# This logic ensures X and y are based on the most up-to-date df state before datetime/text feature engineering on X.
`;


  datetimeExtractions.forEach(ext => {
    advancedPreprocessingInstructions += `
# Datetime Feature Extraction for column '${ext.columnName}'
if '${ext.columnName}' in X.columns and pd.api.types.is_datetime64_any_dtype(X['${ext.columnName}']):`;
    ext.features.forEach(feature => {
      const newColName = `${ext.columnName}_${feature}`;
      advancedPreprocessingInstructions += `
    try:
        if '${feature}' == 'year': X['${newColName}'] = X['${ext.columnName}'].dt.year
        elif '${feature}' == 'month': X['${newColName}'] = X['${ext.columnName}'].dt.month
        elif '${feature}' == 'day': X['${newColName}'] = X['${ext.columnName}'].dt.day
        elif '${feature}' == 'weekday': X['${newColName}'] = X['${ext.columnName}'].dt.weekday
        elif '${feature}' == 'hour': X['${newColName}'] = X['${ext.columnName}'].dt.hour
        print(f"Extracted '${feature}' into '${newColName}'.")
    except AttributeError:
        print(f"Could not extract datetime feature '${feature}' from '${ext.columnName}'. Ensure it's a datetime type.")
`;
    });
    advancedPreprocessingInstructions += `
    # Consider dropping original datetime column after feature extraction if it's not needed as is
    # X = X.drop(columns=['${ext.columnName}']) # Example: X.drop(columns=['${ext.columnName}'], inplace=True, errors='ignore')
    print(f"Note: Original datetime column '${ext.columnName}' is kept. You might want to drop it if it's no longer needed or handle in ColumnTransformer.")
else:
    print(f"Skipping datetime feature extraction for '${ext.columnName}' as it's not a datetime column in X or not present.")
`;
  });
  
  let textFeaturesForColumnTransformer: string[] = [];
  textColumnConfigs.forEach(config => {
      if (config.method !== 'none') {
          textFeaturesForColumnTransformer.push(config.columnName);
      }
  });


  let hyperparameterTuningInstructions = '';
  if (tuningMethod === 'grid_search' || tuningMethod === 'random_search') {
    const searchCV = tuningMethod === 'grid_search' ? 'GridSearchCV' : 'RandomizedSearchCV';
    hyperparameterTuningInstructions = `
10. Hyperparameter Tuning (using ${searchCV}):
   a. Import \`${searchCV}\` from \`sklearn.model_selection\`.
   b. Define a basic parameter grid (for Grid Search) or distribution (for Random Search) appropriate for the '${specificAlgorithmName}' algorithm. 
      - Example for RandomForest: {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
      Provide a small, sensible grid of 1-2 parameters with 2-3 values each for illustration.
   c. Instantiate the search object (e.g., \`${searchCV}(estimator=model, param_grid=param_grid, cv=3, scoring='appropriate_metric_for_task', n_jobs=-1, error_score='raise')\`). Use appropriate scoring (e.g., 'accuracy' for classification, 'neg_mean_squared_error' for regression).
   d. Fit the search object to the training data: \`search.fit(X_train_processed, y_train)\`.
   e. Print the best parameters found: \`print(f"Best parameters found by ${searchCV}: {search.best_params_}")\`.
   f. The main model for prediction and evaluation should now be the best estimator from the search: \`model = search.best_estimator_\`.
`;
  }
  
  let evaluationMetricsImports = '';
  let evaluationMetricsCode = '\n# --- Evaluation Metrics ---';
  if (modelType === 'classification') {
    evaluationMetricsImports += `from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                                confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay)\n`;
    (selectedMetrics as ClassificationMetric[]).forEach(metric => {
      switch (metric) {
        case 'accuracy': evaluationMetricsCode += `print(f"Accuracy: {accuracy_score(y_test_final, y_pred_final):.4f}")\n`; break;
        case 'precision': evaluationMetricsCode += `print(f"Precision: {precision_score(y_test_final, y_pred_final, average='weighted', zero_division=0):.4f}")\n`; break;
        case 'recall': evaluationMetricsCode += `print(f"Recall: {recall_score(y_test_final, y_pred_final, average='weighted', zero_division=0):.4f}")\n`; break;
        case 'f1_score': evaluationMetricsCode += `print(f"F1-score: {f1_score(y_test_final, y_pred_final, average='weighted', zero_division=0):.4f}")\n`; break;
        case 'roc_auc': 
          evaluationMetricsCode += `
if hasattr(model, "predict_proba"):
    try:
        y_pred_proba = model.predict_proba(X_test_processed)
        if y_pred_proba.shape[1] == 2: # Binary classification
            roc_auc = roc_auc_score(y_test_final, y_pred_proba[:, 1])
            print(f"ROC AUC Score: {roc_auc:.4f}")
        else: # Multi-class classification
            roc_auc = roc_auc_score(y_test_final, y_pred_proba, multi_class='ovo', average='weighted')
            print(f"ROC AUC Score (Weighted OVO): {roc_auc:.4f}")
    except Exception as e:
        print(f"Could not calculate ROC AUC Score: {e}")
else:
    print("ROC AUC Score cannot be calculated as the model does not have a 'predict_proba' method.")
`; 
        break;
      }
    });
  } else { // Regression
    evaluationMetricsImports += `from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\nimport numpy as np\n`;
    (selectedMetrics as RegressionMetric[]).forEach(metric => {
      switch (metric) {
        case 'mae': evaluationMetricsCode += `print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_final, y_pred_final):.4f}")\n`; break;
        case 'mse': evaluationMetricsCode += `mse_val = mean_squared_error(y_test_final, y_pred_final)\nprint(f"Mean Squared Error (MSE): {mse_val:.4f}")\n`; break;
        case 'rmse': evaluationMetricsCode += `mse_for_rmse = mean_squared_error(y_test_final, y_pred_final)\nprint(f"Root Mean Squared Error (RMSE): {np.sqrt(mse_for_rmse):.4f}")\n`; break;
        case 'r_squared': evaluationMetricsCode += `print(f"R-squared (R2) Score: {r2_score(y_test_final, y_pred_final):.4f}")\n`; break;
      }
    });
  }

  let visualizationCode = '\n# --- Visualizations ---\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n';
  if (modelType === 'classification') {
    (selectedVisualizations as ClassificationVisualization[]).forEach(viz => {
      if (viz === 'confusion_matrix') {
        visualizationCode += `
# Confusion Matrix
try:
    cm = confusion_matrix(y_test_final, y_pred_final, labels=model.classes_ if hasattr(model, 'classes_') else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_ if hasattr(model, 'classes_') else None)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
except Exception as e:
    print(f"Could not plot Confusion Matrix: {e}")
`;
      } else if (viz === 'roc_curve') {
        visualizationCode += `
# ROC Curve
if hasattr(model, "predict_proba"):
    try:
        y_pred_proba_roc = model.predict_proba(X_test_processed)
        if y_pred_proba_roc.shape[1] == 2: # Binary classification
            fpr, tpr, _ = roc_curve(y_test_final, y_pred_proba_roc[:, 1])
            roc_auc_val = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.show()
        else: 
             print("ROC curve for multi-class scenario requires more specific setup (e.g., one-vs-rest). This basic plot is for binary classification.")
    except Exception as e:
        print(f"Could not plot ROC Curve: {e}")
else:
    print("ROC Curve cannot be plotted as the model does not have a 'predict_proba' method.")
`;
      }
    });
  } else { // Regression
    (selectedVisualizations as RegressionVisualization[]).forEach(viz => {
      if (viz === 'residual_plot') {
        visualizationCode += `
# Residual Plot
try:
    residuals = y_test_final - y_pred_final
    plt.figure()
    sns.scatterplot(x=y_pred_final, y=residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot')
    plt.show()
except Exception as e:
    print(f"Could not plot Residual Plot: {e}")
`;
      }
    });
  }
  if (selectedVisualizations.length === 0) visualizationCode = "";


  // Construct ColumnTransformer parts
  let columnTransformerParts = `
    # Identify feature types from the final X DataFrame (after advanced preprocessing)
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Specified text features for TF-IDF/CountVectorizer
    text_features_for_transformer = ${JSON.stringify(textFeaturesForColumnTransformer)}
    
    # Ensure text features are not also in numeric/categorical lists for default processing
    numeric_features = [f for f in numeric_features if f not in text_features_for_transformer]
    categorical_features = [f for f in categorical_features if f not in text_features_for_transformer]
    
    transformers_list = []
    if len(numeric_features) > 0:
        numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
        transformers_list.append(('num', numeric_pipeline, numeric_features))
    
    if len(categorical_features) > 0:
        categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        transformers_list.append(('cat', categorical_pipeline, categorical_features))
`;

textColumnConfigs.forEach(config => {
    if (config.method === 'tfidf') {
        columnTransformerParts += `
    if '${config.columnName}' in X.columns: # Check if column exists before adding to transformer
        tfidf_pipeline = Pipeline([('tfidf', TfidfVectorizer())])
        transformers_list.append(('text_tfidf_${config.columnName.replace(/[^a-zA-Z0-9_]/g, '_')}', tfidf_pipeline, '${config.columnName}'))
    else:
        print(f"Warning: Text column '${config.columnName}' for TF-IDF not found in X. Skipping in ColumnTransformer.")
`;
    } else if (config.method === 'countvectorizer') {
        columnTransformerParts += `
    if '${config.columnName}' in X.columns: # Check if column exists
        countvec_pipeline = Pipeline([('countvec', CountVectorizer())])
        transformers_list.append(('text_countvec_${config.columnName.replace(/[^a-zA-Z0-9_]/g, '_')}', countvec_pipeline, '${config.columnName}'))
    else:
        print(f"Warning: Text column '${config.columnName}' for CountVectorizer not found in X. Skipping in ColumnTransformer.")
`;
    }
});
columnTransformerParts += `
    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop') # 'drop' unhandled columns. Could be 'passthrough'
`;


  const prompt = `
Generate Python code using scikit-learn (and xgboost if selected) to train a model.
CSV Headers: ${csvHeadersString}. Sample:
\`\`\`csv
${csvSampleString}
\`\`\`
Target: '${targetVariable}'. Model Type: '${modelType}'. Algorithm: '${specificAlgorithmName}'.
Key Choices:
- Outliers: Detection='${outlierMethod}', Handling='${outlierHandling}'.
- Type Conversions: ${JSON.stringify(columnConversions)}.
- Datetime Extraction: ${JSON.stringify(datetimeExtractions)}.
- Text Handling: ${JSON.stringify(textColumnConfigs)}.
- Target Transform: '${targetTransformation}'.
- Tuning: '${tuningMethod}'. Split: ${splitRatio} (test_size=${testSize}), Stratify: '${stratifyOption}'.
- Save Model: ${modelPersistenceOption}. Metrics: ${selectedMetrics.join(', ') || 'None'}. Visuals: ${selectedVisualizations.join(', ') || 'None'}.

Python Code Steps:
1. Imports: pandas, numpy, sklearn (train_test_split, ${specificAlgorithmName}, ${evaluationMetricsImports.trim().replace(/\n/g, ' ')},
   preprocessing: StandardScaler, OneHotEncoder, SimpleImputer, ColumnTransformer, Pipeline),
   text (TfidfVectorizer, CountVectorizer if used), joblib (if saving), matplotlib/seaborn (if visuals).
   XGBoost: \`from xgboost import XGBClassifier/XGBRegressor\`.

2. Load data: \`df = pd.read_csv('your_file.csv')\`. Check target presence.

3. Advanced Preprocessing (on df, or on X, y copies):
   ${advancedPreprocessingInstructions.replace(/\n/g, '\\n')}
   This section defines/redefines X and y. Ensure X and y used from here are the fully processed versions.

4. Final Definition of X and y:
   After all advanced preprocessing, ensure X is the final set of features and y is the final target variable.
   Make sure X does not contain the target variable.

5. Column Preprocessing Setup (ColumnTransformer on final X from step 4):
   ${columnTransformerParts.replace(/\n/g, '\\n')}
   Fit this preprocessor on X_train, then transform X_train and X_test.

6. Split data: \`${trainTestSplitInstruction}\` (using X and y from step 4).

7. Apply Preprocessor:
   \`X_train_processed = preprocessor.fit_transform(X_train)\`
   \`X_test_processed = preprocessor.transform(X_test)\`
   (Optional: Convert X_train_processed, X_test_processed back to DataFrames with \`preprocessor.get_feature_names_out()\`).

8. Initialize Model: \`${specificAlgorithmName}(random_state=42)\`. XGBoost: \`use_label_encoder=False\`, set \`eval_metric\`.

${hyperparameterTuningInstructions.replace("10.", "9.")} <!-- Adjust step number -->

10. Train Model: (If tuning, model is \`search.best_estimator_\`). Else: \`model.fit(X_train_processed, y_train)\`.

11. Predictions: \`y_pred = model.predict(X_test_processed)\`.

12. Inverse Transform (if target was transformed, e.g. 'log_transform'):
    Define \`y_pred_final\` and \`y_test_final\`.
    If '${targetTransformation}' == 'log_transform' AND modelType is 'regression':
        \`y_pred_final = np.expm1(y_pred)\`
        \`y_test_final = np.expm1(y_test)\`
    Else:
        \`y_pred_final = y_pred\`
        \`y_test_final = y_test\`
    Use these \`_final\` versions for metrics and relevant visualizations.

13. Evaluation Metrics:
    ${evaluationMetricsCode.trim().replace(/\n/g, '\\n')} <!-- Use y_test_final, y_pred_final -->

14. Visualizations:
    ${visualizationCode.trim().replace(/\n/g, '\\n')} <!-- Use y_test_final, y_pred_final -->

15. ${modelPersistenceInstruction}

Output ONLY Python code. Ensure runnable after 'your_file.csv' is set.
Gracefully handle cases like predict_proba for models without it.
Group all imports at the top.
For ColumnTransformer, \`remainder='drop'\` is generally safer unless unhandled columns are desired.
Ensure correct X and y are used at each stage (raw, after type conversion, after outlier removal, after feature engineering). The sequence of operations in advanced_preprocessing is critical. Outlier removal should use data that has correct types. Type conversion is on 'df', then X/y are derived. Outlier removal operates on these X/y. Then further feature engineering on X.
`;

  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
      model: TEXT_MODEL_NAME,
      contents: prompt,
      config: { temperature: 0.45 } 
    });

    let code = response.text.trim();
    const fenceRegex = /^```(?:python)?\s*\n?(.*?)\n?\s*```$/s;
    const match = code.match(fenceRegex);
    if (match && match[1]) {
      code = match[1].trim();
    }
    
    return code;
  } catch (error) {
    console.error("Error generating Python model code:", error);
    throw new Error(`Failed to generate Python code from AI: ${error instanceof Error ? error.message : String(error)}`);
  }
};

export const generateCorrelationMatrixCode = async (
  headers: string[],
  targetVariable: string | null
): Promise<string> => {
  if (!ai) throw new Error("Gemini API Client not initialized. API_KEY might be missing.");

  const numericInstructions = targetVariable 
    ? `Consider which columns are likely numeric. The target variable is '${targetVariable}'. The correlation matrix should ideally show correlations between numeric features and also with the target variable if it's numeric.`
    : `Consider which columns are likely numeric.`;

  const prompt = `
Generate Python code to create and display a correlation matrix heatmap for a dataset.
The dataset has the following headers: ${JSON.stringify(headers)}.
${numericInstructions}

The Python code should:
1. Import necessary libraries: pandas, numpy, seaborn, and matplotlib.pyplot.
2. Include a placeholder for loading the full CSV into a pandas DataFrame, e.g., \`df = pd.read_csv('your_file.csv')\`.
3. Select only numeric columns from the DataFrame for the correlation calculation (\`df.select_dtypes(include=np.number)\`). Import numpy as np.
4. Calculate the correlation matrix for these numeric columns.
5. Use seaborn to create a heatmap of the correlation matrix. Include annotations (\`annot=True\`), a colormap (e.g., \`cmap='coolwarm'\`), and format annotations to two decimal places (\`fmt=".2f"\`).
6. Use matplotlib.pyplot to display the heatmap (\`plt.title('Correlation Heatmap of Numeric Features')\`, \`plt.show()\`). Ensure the plot is well-titled.
7. Include comments to explain the steps.
8. If '${targetVariable}' is specified and likely numeric, ensure its correlations are visible. If it's categorical, mention that it won't be in the numeric correlation matrix in a comment.

Provide ONLY the Python code block, without any surrounding text or explanations. The code should be ready to be copied and run after replacing 'your_file.csv'.
Ensure all necessary imports are at the top of the script.
`;

  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
      model: TEXT_MODEL_NAME,
      contents: prompt,
      config: { temperature: 0.3 }
    });

    let code = response.text.trim();
    const fenceRegex = /^```(?:python)?\s*\n?(.*?)\n?\s*```$/s;
    const match = code.match(fenceRegex);
    if (match && match[1]) {
      code = match[1].trim();
    }
    return code;
  } catch (error) {
    console.error("Error generating correlation matrix code:", error);
    throw new Error(`Failed to generate correlation matrix code from AI: ${error instanceof Error ? error.message : String(error)}`);
  }
};
