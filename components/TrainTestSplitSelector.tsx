import React from 'react';
import { ModelType, TrainTestSplitRatio, StratifyOption } from '../types';

interface TrainTestSplitSelectorProps {
  modelType: ModelType;
  selectedRatio: TrainTestSplitRatio;
  onRatioChange: (ratio: TrainTestSplitRatio) => void;
  selectedStratify: StratifyOption;
  onStratifyChange: (option: StratifyOption) => void;
  onProceed: () => void;
}

const ratioOptions: { id: TrainTestSplitRatio; label: string; description: string }[] = [
  { id: '80/20', label: '80% Train / 20% Test', description: 'A common split, allocates 80% of data for training.' },
  { id: '75/25', label: '75% Train / 25% Test', description: 'Allocates 75% for training, slightly larger test set.' },
  { id: '70/30', label: '70% Train / 30% Test', description: 'Allocates 70% for training, useful for larger datasets or more robust testing.' },
];

const stratifyOptions: { id: StratifyOption; label: string; description: string }[] = [
    { id: 'yes', label: 'Yes, use Stratification', description: 'Ensures train and test sets have proportional class representation. Recommended for classification.' },
    { id: 'no', label: 'No Stratification', description: 'Randomly splits data without considering class proportions.' },
];

export const TrainTestSplitSelector: React.FC<TrainTestSplitSelectorProps> = ({
  modelType,
  selectedRatio,
  onRatioChange,
  selectedStratify,
  onStratifyChange,
  onProceed,
}) => {
  return (
    <div className="space-y-8 p-6 bg-slate-700 rounded-lg shadow-lg">
      <div>
        <h3 className="text-xl font-semibold text-sky-400 mb-1">Train/Test Split Ratio:</h3>
        <p className="text-sm text-slate-400 mb-4">
          This is step 7. Choose how to split your data for training and testing the model.
        </p>
        <fieldset className="space-y-3">
          <legend className="sr-only">Train/Test Split Ratio</legend>
          {ratioOptions.map((option) => (
            <div key={option.id} className="relative flex items-start p-3 rounded-md border border-slate-600 hover:border-sky-500 transition-colors bg-slate-700/50 has-[:checked]:bg-sky-500/10 has-[:checked]:border-sky-500">
              <div className="flex h-6 items-center">
                <input
                  id={`ratio-${option.id}`}
                  name="trainTestSplitRatio"
                  type="radio"
                  checked={selectedRatio === option.id}
                  onChange={() => onRatioChange(option.id)}
                  className="h-5 w-5 text-sky-500 border-slate-500 focus:ring-sky-400 bg-slate-600 cursor-pointer"
                  aria-describedby={`ratio-${option.id}-description`}
                />
              </div>
              <div className="ml-3 text-sm leading-6">
                <label
                  htmlFor={`ratio-${option.id}`}
                  className="block text-md font-medium text-slate-100 cursor-pointer"
                >
                  {option.label}
                </label>
                <p id={`ratio-${option.id}-description`} className="text-slate-400 text-xs">
                  {option.description}
                </p>
              </div>
            </div>
          ))}
        </fieldset>
      </div>

      {modelType === 'classification' && (
        <div>
          <h3 className="text-xl font-semibold text-sky-400 mb-1">Stratification:</h3>
          <p className="text-sm text-slate-400 mb-4">
            For classification tasks, stratification ensures that both train and test sets have similar proportions of target classes.
          </p>
          <fieldset className="space-y-3">
            <legend className="sr-only">Stratification Option</legend>
            {stratifyOptions.map((option) => (
              <div key={option.id} className="relative flex items-start p-3 rounded-md border border-slate-600 hover:border-sky-500 transition-colors bg-slate-700/50 has-[:checked]:bg-sky-500/10 has-[:checked]:border-sky-500">
                <div className="flex h-6 items-center">
                  <input
                    id={`stratify-${option.id}`}
                    name="stratifyOption"
                    type="radio"
                    checked={selectedStratify === option.id}
                    onChange={() => onStratifyChange(option.id)}
                    className="h-5 w-5 text-sky-500 border-slate-500 focus:ring-sky-400 bg-slate-600 cursor-pointer"
                    aria-describedby={`stratify-${option.id}-description`}
                  />
                </div>
                <div className="ml-3 text-sm leading-6">
                  <label
                    htmlFor={`stratify-${option.id}`}
                    className="block text-md font-medium text-slate-100 cursor-pointer"
                  >
                    {option.label}
                  </label>
                   <p id={`stratify-${option.id}-description`} className="text-slate-400 text-xs">
                    {option.description}
                  </p>
                </div>
              </div>
            ))}
          </fieldset>
        </div>
      )}

      <button
        onClick={onProceed}
        disabled={!selectedRatio || (modelType === 'classification' && !selectedStratify)}
        className="w-full bg-sky-500 hover:bg-sky-600 disabled:bg-slate-500 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded-lg shadow-md transition duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-opacity-50"
        aria-label="Proceed to model persistence option"
      >
        Proceed to Model Persistence Option
      </button>
    </div>
  );
};