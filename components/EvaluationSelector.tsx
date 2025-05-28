
import React from 'react';
import { 
    ModelType, ClassificationMetric, RegressionMetric,
    ClassificationVisualization, RegressionVisualization
} from '../types';

interface EvaluationSelectorProps {
  modelType: ModelType;
  selectedClassificationMetrics: ClassificationMetric[];
  onClassificationMetricsChange: (metrics: ClassificationMetric[]) => void;
  selectedRegressionMetrics: RegressionMetric[];
  onRegressionMetricsChange: (metrics: RegressionMetric[]) => void;
  selectedClassificationVisualizations: ClassificationVisualization[];
  onClassificationVisualizationsChange: (visualizations: ClassificationVisualization[]) => void;
  selectedRegressionVisualizations: RegressionVisualization[];
  onRegressionVisualizationsChange: (visualizations: RegressionVisualization[]) => void;
  onSubmit: () => void;
}

const classificationMetricsOptions: { id: ClassificationMetric; label: string; description: string }[] = [
  { id: 'accuracy', label: 'Accuracy', description: 'Overall correctness of the model.' },
  { id: 'precision', label: 'Precision', description: 'Ability of the classifier not to label as positive a sample that is negative.' },
  { id: 'recall', label: 'Recall (Sensitivity)', description: 'Ability of the classifier to find all the positive samples.' },
  { id: 'f1_score', label: 'F1-Score', description: 'Weighted average of Precision and Recall.' },
  { id: 'roc_auc', label: 'ROC AUC', description: 'Area Under the Receiver Operating Characteristic Curve; ability to distinguish between classes.' },
];

const regressionMetricsOptions: { id: RegressionMetric; label: string; description: string }[] = [
  { id: 'mae', label: 'Mean Absolute Error (MAE)', description: 'Average absolute difference between predicted and actual values.' },
  { id: 'mse', label: 'Mean Squared Error (MSE)', description: 'Average squared difference between predicted and actual values.' },
  { id: 'rmse', label: 'Root Mean Squared Error (RMSE)', description: 'Square root of MSE, in the same units as the target.' },
  { id: 'r_squared', label: 'R-squared (R²)', description: 'Proportion of the variance in the dependent variable predictable from the independent variables.' },
];

const classificationVisualizationsOptions: { id: ClassificationVisualization; label: string; description: string }[] = [
  { id: 'confusion_matrix', label: 'Confusion Matrix', description: 'Table showing true vs. predicted classifications.' },
  { id: 'roc_curve', label: 'ROC Curve', description: 'Plot illustrating diagnostic ability of a binary classifier as its discrimination threshold is varied.' },
];

const regressionVisualizationsOptions: { id: RegressionVisualization; label: string; description: string }[] = [
  { id: 'residual_plot', label: 'Residual Plot', description: 'Scatter plot of residuals vs. predicted values to check assumptions.' },
];


export const EvaluationSelector: React.FC<EvaluationSelectorProps> = ({
  modelType,
  selectedClassificationMetrics,
  onClassificationMetricsChange,
  selectedRegressionMetrics,
  onRegressionMetricsChange,
  selectedClassificationVisualizations,
  onClassificationVisualizationsChange,
  selectedRegressionVisualizations,
  onRegressionVisualizationsChange,
  onSubmit,
}) => {

  const handleClassificationMetricChange = (metricId: ClassificationMetric) => {
    const currentSelection = [...selectedClassificationMetrics];
    const index = currentSelection.indexOf(metricId);
    if (index > -1) {
      currentSelection.splice(index, 1);
    } else {
      currentSelection.push(metricId);
    }
    onClassificationMetricsChange(currentSelection);
  };

  const handleRegressionMetricChange = (metricId: RegressionMetric) => {
    const currentSelection = [...selectedRegressionMetrics];
    const index = currentSelection.indexOf(metricId);
    if (index > -1) {
      currentSelection.splice(index, 1);
    } else {
      currentSelection.push(metricId);
    }
    onRegressionMetricsChange(currentSelection);
  };

  const handleClassificationVisualizationChange = (vizId: ClassificationVisualization) => {
    const currentSelection = [...selectedClassificationVisualizations];
    const index = currentSelection.indexOf(vizId);
    if (index > -1) {
      currentSelection.splice(index, 1);
    } else {
      currentSelection.push(vizId);
    }
    onClassificationVisualizationsChange(currentSelection);
  };

  const handleRegressionVisualizationChange = (vizId: RegressionVisualization) => {
    const currentSelection = [...selectedRegressionVisualizations];
    const index = currentSelection.indexOf(vizId);
    if (index > -1) {
      currentSelection.splice(index, 1);
    } else {
      currentSelection.push(vizId);
    }
    onRegressionVisualizationsChange(currentSelection);
  };
  
  const isSubmitDisabled = modelType === 'classification' 
    ? (selectedClassificationMetrics.length === 0 && selectedClassificationVisualizations.length === 0)
    : (selectedRegressionMetrics.length === 0 && selectedRegressionVisualizations.length === 0);

  return (
    <div className="space-y-8 p-6 bg-slate-700 rounded-lg shadow-lg">
      {modelType === 'classification' ? (
        <>
          <div>
            <h3 className="text-xl font-semibold text-sky-400 mb-1">Select Classification Metrics:</h3>
            <p className="text-sm text-slate-400 mb-4">
              This is step 9. Choose the metrics to evaluate your 'classification' model. The generated Python code will calculate and print these.
            </p>
            <fieldset className="space-y-3">
              <legend className="sr-only">Classification Metrics</legend>
              {classificationMetricsOptions.map((metricOption) => (
                <div key={metricOption.id} className="relative flex items-start p-3 rounded-md border border-slate-600 hover:border-sky-500 transition-colors bg-slate-700/50 has-[:checked]:bg-sky-500/10 has-[:checked]:border-sky-500">
                  <div className="flex h-6 items-center">
                    <input
                      id={`metric-${metricOption.id}`}
                      name="classificationMetrics"
                      type="checkbox"
                      checked={selectedClassificationMetrics.includes(metricOption.id)}
                      onChange={() => handleClassificationMetricChange(metricOption.id)}
                      className="h-5 w-5 text-sky-500 border-slate-500 focus:ring-sky-400 bg-slate-600 rounded cursor-pointer"
                      aria-describedby={`metric-${metricOption.id}-description`}
                    />
                  </div>
                  <div className="ml-3 text-sm leading-6">
                    <label
                      htmlFor={`metric-${metricOption.id}`}
                      className="block text-md font-medium text-slate-100 cursor-pointer"
                    >
                      {metricOption.label}
                    </label>
                    <p id={`metric-${metricOption.id}-description`} className="text-slate-400 text-xs">
                      {metricOption.description}
                    </p>
                  </div>
                </div>
              ))}
            </fieldset>
          </div>

          <div>
            <h3 className="text-xl font-semibold text-sky-400 mb-1">Select Classification Visualizations:</h3>
            <p className="text-sm text-slate-400 mb-4">
              Choose visualizations to include in the Python code for deeper model analysis.
            </p>
            <fieldset className="space-y-3">
              <legend className="sr-only">Classification Visualizations</legend>
              {classificationVisualizationsOptions.map((vizOption) => (
                <div key={vizOption.id} className="relative flex items-start p-3 rounded-md border border-slate-600 hover:border-sky-500 transition-colors bg-slate-700/50 has-[:checked]:bg-sky-500/10 has-[:checked]:border-sky-500">
                  <div className="flex h-6 items-center">
                    <input
                      id={`viz-${vizOption.id}`}
                      name="classificationVisualizations"
                      type="checkbox"
                      checked={selectedClassificationVisualizations.includes(vizOption.id)}
                      onChange={() => handleClassificationVisualizationChange(vizOption.id)}
                      className="h-5 w-5 text-sky-500 border-slate-500 focus:ring-sky-400 bg-slate-600 rounded cursor-pointer"
                      aria-describedby={`viz-${vizOption.id}-description`}
                    />
                  </div>
                  <div className="ml-3 text-sm leading-6">
                    <label
                      htmlFor={`viz-${vizOption.id}`}
                      className="block text-md font-medium text-slate-100 cursor-pointer"
                    >
                      {vizOption.label}
                    </label>
                    <p id={`viz-${vizOption.id}-description`} className="text-slate-400 text-xs">
                      {vizOption.description}
                    </p>
                  </div>
                </div>
              ))}
            </fieldset>
          </div>
        </>
      ) : ( // modelType === 'regression'
        <>
          <div>
            <h3 className="text-xl font-semibold text-sky-400 mb-1">Select Regression Metrics:</h3>
            <p className="text-sm text-slate-400 mb-4">
              This is step 9. Choose the metrics to evaluate your 'regression' model. The generated Python code will calculate and print these.
            </p>
            <fieldset className="space-y-3">
              <legend className="sr-only">Regression Metrics</legend>
              {regressionMetricsOptions.map((metricOption) => (
                <div key={metricOption.id} className="relative flex items-start p-3 rounded-md border border-slate-600 hover:border-sky-500 transition-colors bg-slate-700/50 has-[:checked]:bg-sky-500/10 has-[:checked]:border-sky-500">
                  <div className="flex h-6 items-center">
                    <input
                      id={`metric-${metricOption.id}`}
                      name="regressionMetrics"
                      type="checkbox"
                      checked={selectedRegressionMetrics.includes(metricOption.id)}
                      onChange={() => handleRegressionMetricChange(metricOption.id)}
                      className="h-5 w-5 text-sky-500 border-slate-500 focus:ring-sky-400 bg-slate-600 rounded cursor-pointer"
                      aria-describedby={`metric-${metricOption.id}-description`}
                    />
                  </div>
                  <div className="ml-3 text-sm leading-6">
                    <label
                      htmlFor={`metric-${metricOption.id}`}
                      className="block text-md font-medium text-slate-100 cursor-pointer"
                    >
                      {metricOption.label}
                    </label>
                    <p id={`metric-${metricOption.id}-description`} className="text-slate-400 text-xs">
                      {metricOption.description}
                    </p>
                  </div>
                </div>
              ))}
            </fieldset>
          </div>

          <div>
            <h3 className="text-xl font-semibold text-sky-400 mb-1">Select Regression Visualizations:</h3>
            <p className="text-sm text-slate-400 mb-4">
              Choose visualizations to include in the Python code for deeper model analysis.
            </p>
            <fieldset className="space-y-3">
              <legend className="sr-only">Regression Visualizations</legend>
              {regressionVisualizationsOptions.map((vizOption) => (
                <div key={vizOption.id} className="relative flex items-start p-3 rounded-md border border-slate-600 hover:border-sky-500 transition-colors bg-slate-700/50 has-[:checked]:bg-sky-500/10 has-[:checked]:border-sky-500">
                  <div className="flex h-6 items-center">
                    <input
                      id={`viz-${vizOption.id}`}
                      name="regressionVisualizations"
                      type="checkbox"
                      checked={selectedRegressionVisualizations.includes(vizOption.id)}
                      onChange={() => handleRegressionVisualizationChange(vizOption.id)}
                      className="h-5 w-5 text-sky-500 border-slate-500 focus:ring-sky-400 bg-slate-600 rounded cursor-pointer"
                      aria-describedby={`viz-${vizOption.id}-description`}
                    />
                  </div>
                  <div className="ml-3 text-sm leading-6">
                    <label
                      htmlFor={`viz-${vizOption.id}`}
                      className="block text-md font-medium text-slate-100 cursor-pointer"
                    >
                      {vizOption.label}
                    </label>
                    <p id={`viz-${vizOption.id}-description`} className="text-slate-400 text-xs">
                      {vizOption.description}
                    </p>
                  </div>
                </div>
              ))}
            </fieldset>
          </div>
        </>
      )}

      <button
        onClick={onSubmit}
        disabled={isSubmitDisabled}
        className="w-full bg-green-500 hover:bg-green-600 disabled:bg-slate-500 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded-lg shadow-md transition duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50"
        aria-label="Generate Python model code with selected evaluations"
      >
        Generate Python Model Code
      </button>
       {isSubmitDisabled && (
        <p className="text-xs text-amber-400 text-center mt-2">
          Please select at least one metric or visualization to proceed.
        </p>
      )}
    </div>
  );
};