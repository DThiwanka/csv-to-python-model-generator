import React from 'react';
import { ModelPersistenceOption } from '../types';

interface ModelPersistenceSelectorProps {
  selectedOption: ModelPersistenceOption | null;
  onOptionSelect: (option: ModelPersistenceOption) => void;
  onProceed: () => void; 
}

export const ModelPersistenceSelector: React.FC<ModelPersistenceSelectorProps> = ({
  selectedOption,
  onOptionSelect,
  onProceed, 
}) => {
  const options: { id: ModelPersistenceOption; label: string; description: string }[] = [
    { 
      id: 'yes', 
      label: 'Yes, save the model',
      description: 'Include code to save the trained model to a file (e.g., using joblib).'
    },
    { 
      id: 'no', 
      label: 'No, don\'t save',
      description: 'Do not include code for saving the model to a file.'
    },
  ];

  return (
    <div className="space-y-6 p-6 bg-slate-700 rounded-lg shadow-lg">
      <div>
        <h3 className="text-xl font-semibold text-sky-400 mb-1">Save Trained Model?</h3>
        <p className="text-sm text-slate-400 mb-4">This is step 8. Would you like the Python script to include code for saving the trained model to a file?</p>
        <fieldset className="space-y-4">
          <legend className="sr-only">Model persistence option</legend>
          {options.map((option) => (
            <div key={option.id} className="relative flex items-start p-3 rounded-md border border-slate-600 hover:border-sky-500 transition-colors bg-slate-700/50 has-[:checked]:bg-sky-500/10 has-[:checked]:border-sky-500">
              <div className="flex h-6 items-center">
                <input
                  id={option.id}
                  name="modelPersistence"
                  type="radio"
                  checked={selectedOption === option.id}
                  onChange={() => onOptionSelect(option.id)}
                  className="h-5 w-5 text-sky-500 border-slate-500 focus:ring-sky-400 bg-slate-600 cursor-pointer"
                  aria-describedby={`${option.id}-description`}
                />
              </div>
              <div className="ml-3 text-sm leading-6">
                <label
                  htmlFor={option.id}
                  className="block text-md font-medium text-slate-100 cursor-pointer"
                >
                  {option.label}
                </label>
                <p id={`${option.id}-description`} className="text-slate-400 text-xs">
                  {option.description}
                </p>
              </div>
            </div>
          ))}
        </fieldset>
      </div>

      <button
        onClick={onProceed} 
        disabled={!selectedOption}
        className="w-full bg-sky-500 hover:bg-sky-600 disabled:bg-slate-500 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded-lg shadow-md transition duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-opacity-50"
        aria-label="Proceed to evaluation options" 
      >
        Proceed to Evaluation Options 
      </button>
    </div>
  );
};