
import React from 'react';
import { ModelType } from '../types';

interface ModelTypeSelectorProps {
  selectedModelType: ModelType | null;
  onModelTypeSelect: (modelType: ModelType) => void;
  onProceed: () => void; 
}

export const ModelTypeSelector: React.FC<ModelTypeSelectorProps> = ({
  selectedModelType,
  onModelTypeSelect,
  onProceed,
}) => {
  const modelTypes: { id: ModelType; label: string; description: string }[] = [
    { id: 'classification', label: 'Classification', description: 'Predicting categories or discrete classes (e.g., spam/not spam, cat/dog).' },
    { id: 'regression', label: 'Regression', description: 'Predicting continuous numerical values (e.g., house price, temperature).' },
  ];

  return (
    <div className="space-y-6 p-6 bg-slate-700 rounded-lg shadow-lg">
      <div>
        <h3 className="text-xl font-semibold text-sky-400 mb-1">Choose Model Type:</h3>
        <p className="text-sm text-slate-400 mb-4">This is step 5 of the process. Select the general type of prediction task. This will filter the available algorithms and evaluation options in subsequent steps.</p>
        <fieldset className="space-y-4">
          <legend className="sr-only">Model type</legend>
          {modelTypes.map((modelTypeOption) => (
             <div key={modelTypeOption.id} className="relative flex items-start p-3 rounded-md border border-slate-600 hover:border-sky-500 transition-colors bg-slate-700/50 has-[:checked]:bg-sky-500/10 has-[:checked]:border-sky-500">
                <div className="flex h-6 items-center">
                    <input
                    id={modelTypeOption.id}
                    name="modelType"
                    type="radio"
                    checked={selectedModelType === modelTypeOption.id}
                    onChange={() => onModelTypeSelect(modelTypeOption.id)}
                    className="h-5 w-5 text-sky-500 border-slate-500 focus:ring-sky-400 bg-slate-600 cursor-pointer"
                    aria-describedby={`${modelTypeOption.id}-description`}
                    />
                </div>
                <div className="ml-3 text-sm leading-6">
                    <label
                    htmlFor={modelTypeOption.id}
                    className="block text-md font-medium text-slate-100 cursor-pointer"
                    >
                    {modelTypeOption.label}
                    </label>
                    <p id={`${modelTypeOption.id}-description`} className="text-slate-400 text-xs">
                        {modelTypeOption.description}
                    </p>
                </div>
            </div>
          ))}
        </fieldset>
      </div>

      <button
        onClick={onProceed} 
        disabled={!selectedModelType}
        className="w-full bg-sky-500 hover:bg-sky-600 disabled:bg-slate-500 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded-lg shadow-md transition duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-opacity-50"
        aria-label="Proceed to algorithm selection"
      >
        Proceed to Algorithm Selection
      </button>
    </div>
  );
};
