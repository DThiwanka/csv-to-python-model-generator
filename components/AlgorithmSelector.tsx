import React from 'react';
import { ModelType, Algorithm, ClassificationAlgorithm, RegressionAlgorithm } from '../types';

interface AlgorithmSelectorProps {
  modelType: ModelType;
  selectedAlgorithm: Algorithm | null;
  onAlgorithmSelect: (algorithm: Algorithm) => void;
  onProceed: () => void;
}

const classificationAlgorithms: { id: ClassificationAlgorithm; label: string; description: string }[] = [
  { id: 'logistic_regression', label: 'Logistic Regression', description: 'Good for binary classification, interpretable.' },
  { id: 'random_forest_classifier', label: 'Random Forest Classifier', description: 'Ensemble method, robust, handles non-linearity well.' },
  { id: 'svm_classifier', label: 'Support Vector Machine (SVM)', description: 'Effective in high dimensional spaces, can be kernelized.' },
  { id: 'gradient_boosting_classifier', label: 'Gradient Boosting Classifier', description: 'Powerful ensemble, builds trees sequentially.' },
  { id: 'knn_classifier', label: 'K-Nearest Neighbors (KNN)', description: 'Simple instance-based learner, sensitive to feature scaling.' },
  { id: 'xgboost_classifier', label: 'XGBoost Classifier', description: 'Optimized distributed gradient boosting, often high performance.' },
];

const regressionAlgorithms: { id: RegressionAlgorithm; label: string; description: string }[] = [
  { id: 'linear_regression', label: 'Linear Regression', description: 'Basic regression, assumes linear relationship.' },
  { id: 'random_forest_regressor', label: 'Random Forest Regressor', description: 'Ensemble method, good for complex relationships.' },
  { id: 'svm_regressor', label: 'Support Vector Regressor (SVR)', description: 'SVM adapted for regression tasks.' },
  { id: 'gradient_boosting_regressor', label: 'Gradient Boosting Regressor', description: 'Powerful ensemble for regression.' },
  { id: 'knn_regressor', label: 'K-Nearest Neighbors (KNN) Regressor', description: 'Predicts by averaging neighbors, sensitive to scaling.' },
  { id: 'xgboost_regressor', label: 'XGBoost Regressor', description: 'High-performance gradient boosting for regression.' },
];

export const AlgorithmSelector: React.FC<AlgorithmSelectorProps> = ({
  modelType,
  selectedAlgorithm,
  onAlgorithmSelect,
  onProceed,
}) => {
  const algorithms = modelType === 'classification' ? classificationAlgorithms : regressionAlgorithms;

  return (
    <div className="space-y-6 p-6 bg-slate-700 rounded-lg shadow-lg">
      <div>
        <h3 className="text-xl font-semibold text-sky-400 mb-1">Choose Specific Algorithm:</h3>
        <p className="text-sm text-slate-400 mb-4">
          This is step 5. Based on your selection of '{modelType}', choose a specific algorithm.
        </p>
        <fieldset className="space-y-4">
          <legend className="sr-only">Algorithm selection</legend>
          {algorithms.map((algoOption) => (
            <div key={algoOption.id} className="relative flex items-start p-3 rounded-md border border-slate-600 hover:border-sky-500 transition-colors bg-slate-700/50 has-[:checked]:bg-sky-500/10 has-[:checked]:border-sky-500">
              <div className="flex h-6 items-center">
                <input
                  id={algoOption.id}
                  name="algorithm"
                  type="radio"
                  checked={selectedAlgorithm === algoOption.id}
                  onChange={() => onAlgorithmSelect(algoOption.id as Algorithm)}
                  className="h-5 w-5 text-sky-500 border-slate-500 focus:ring-sky-400 bg-slate-600 cursor-pointer"
                  aria-describedby={`${algoOption.id}-description`}
                />
              </div>
              <div className="ml-3 text-sm leading-6">
                <label
                  htmlFor={algoOption.id}
                  className="block text-md font-medium text-slate-100 cursor-pointer"
                >
                  {algoOption.label}
                </label>
                <p id={`${algoOption.id}-description`} className="text-slate-400 text-xs">
                  {algoOption.description}
                </p>
              </div>
            </div>
          ))}
        </fieldset>
      </div>

      <button
        onClick={onProceed}
        disabled={!selectedAlgorithm}
        className="w-full bg-sky-500 hover:bg-sky-600 disabled:bg-slate-500 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded-lg shadow-md transition duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-opacity-50"
        aria-label="Proceed to hyperparameter tuning options"
      >
        Proceed to Hyperparameter Tuning
      </button>
    </div>
  );
};