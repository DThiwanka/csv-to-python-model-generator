export interface CSVRow {
  [key: string]: string | number | null | undefined; // Allow null or undefined for missing values
}

export type ModelType = 'classification' | 'regression';

export type ClassificationAlgorithm = 
  | 'logistic_regression' 
  | 'random_forest_classifier' 
  | 'svm_classifier' // Support Vector Machine for Classification
  | 'gradient_boosting_classifier'
  | 'knn_classifier' // K-Nearest Neighbors
  | 'xgboost_classifier';

export type RegressionAlgorithm = 
  | 'linear_regression' 
  | 'random_forest_regressor' 
  | 'svm_regressor' // Support Vector Machine for Regression
  | 'gradient_boosting_regressor'
  | 'knn_regressor'
  | 'xgboost_regressor';

export type Algorithm = ClassificationAlgorithm | RegressionAlgorithm;

export type HyperparameterTuningMethod = 'none' | 'grid_search' | 'random_search';

export type TrainTestSplitRatio = '80/20' | '75/25' | '70/30';
export type StratifyOption = 'yes' | 'no';

export type ModelPersistenceOption = 'yes' | 'no';

export type ClassificationMetric = 'accuracy' | 'precision' | 'recall' | 'f1_score' | 'roc_auc';
export type RegressionMetric = 'mae' | 'mse' | 'rmse' | 'r_squared';
export type EvaluationMetric = ClassificationMetric | RegressionMetric;

export type ClassificationVisualization = 'confusion_matrix' | 'roc_curve';
export type RegressionVisualization = 'residual_plot';
export type EvaluationVisualization = ClassificationVisualization | RegressionVisualization;

// Advanced Preprocessing Types
export type OutlierMethod = 'none' | 'z_score' | 'iqr';
export type OutlierHandling = 'none' | 'remove'; // Future: 'cap', 'impute'

export type ForcedColumnType = 'numeric' | 'categorical' | 'datetime' | 'text';
export interface ColumnConversionConfig {
  columnName: string;
  convertTo: ForcedColumnType;
}

export type DatetimeFeature = 'year' | 'month' | 'day' | 'weekday' | 'hour';
export interface DatetimeExtractionConfig {
  columnName: string;
  features: DatetimeFeature[];
}

export type TextHandlingMethod = 'none' | 'tfidf' | 'countvectorizer';
export interface TextColumnConfig {
  columnName: string;
  method: TextHandlingMethod;
}

export type TargetTransformationMethod = 'none' | 'log_transform';


// Step 1: initial
// Step 2: file_processing
// Step 3: target_selection
// Step 4: data_exploration
// Step 5: advanced_preprocessing_selection // NEW STEP
// Step 6: model_type_selection (Classif/Regress) // Was 5
// Step 7: algorithm_selection (Specific algo) // Was 6
// Step 8: hyperparameter_tuning_selection // Was 7
// Step 9: train_test_split_options // Was 8
// Step 10: model_persistence_selection // Was 9
// Step 11: evaluation_selection  // Was 10
// Step 12: code_generation     // Was 11
// Step 13: code_display        // Was 12
// Step 14: error               // Was 13
export type AppStep = 
  | 'initial' 
  | 'file_processing' 
  | 'target_selection' 
  | 'data_exploration'
  | 'advanced_preprocessing_selection' // New
  | 'model_type_selection'
  | 'algorithm_selection' 
  | 'hyperparameter_tuning_selection'
  | 'train_test_split_options'
  | 'model_persistence_selection'
  | 'evaluation_selection' 
  | 'code_generation' 
  | 'code_display' 
  | 'error';

export interface NumericSummary {
  type: 'numeric';
  count: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  nulls: number;
}

export interface CategoricalSummary {
  type: 'categorical';
  count: number;
  unique: number;
  top: string | number | null;
  freq: number | null;
  nulls: number;
}

export type ColumnSummary = NumericSummary | CategoricalSummary;

export interface SummaryStats {
  [column: string]: ColumnSummary;
}

export interface MissingValueInfo {
  column: string;
  missingCount: number;
  totalCount: number;
  percentage: number;
}
