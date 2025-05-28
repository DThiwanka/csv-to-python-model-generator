import React from 'react';
import { Algorithm, HyperparameterTuningMethod } from '../types';

interface HyperparameterTuningSelectorProps {
  selectedAlgorithm: Algorithm; 
  selectedTuningMethod: HyperparameterTuningMethod | null;
  onTuningMethodSelect: (method: HyperparameterTuningMethod) => void;
  onProceed: () => void;
}

export const HyperparameterTuningSelector: React.FC<HyperparameterTuningSelectorProps> = ({
  selectedAlgorithm,
  selectedTuningMethod,
  onTuningMethodSelect,
  onProceed,
}) => {
  const tuningOptions: { id: HyperparameterTuningMethod; label: string; description: string }[] = [
    { 
      id: 'none', 
      label: 'None', 
      description: 'Use default hyperparameters for the selected algorithm. Fastest option.' 
    },
    { 
      id: 'grid_search', 
      label: 'Grid Search', 
      description: 'Exhaustively searches a predefined set of hyperparameter combinations. Can be slow.' 
    },
    { 
      id: 'random_search', 
      label: 'Random Search', 
      description: 'Samples a fixed number of hyperparameter combinations from specified distributions. Often more efficient than Grid Search.' 
    },
  ];
  // Format algorithm name for display
  const formattedAlgorithmName = selectedAlgorithm
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');


  return (
    <div className="space-y-6 p-6 bg-slate-700 rounded-lg shadow-lg">
      <div>
        <h3 className="text-xl font-semibold text-sky-400 mb-1">Hyperparameter Tuning Option:</h3>
        <p className="text-sm text-slate-400 mb-4">
          This is step 6. For the selected algorithm ({formattedAlgorithmName}), choose if and how to tune hyperparameters.
        </p>
        <fieldset className="space-y-4">
          <legend className="sr-only">Hyperparameter tuning method</legend>
          {tuningOptions.map((option) => (
            <div key={option.id} className="relative flex items-start p-3 rounded-md border border-slate-600 hover:border-sky-500 transition-colors bg-slate-700/50 has-[:checked]:bg-sky-500/10 has-[:checked]:border-sky-500">
              <div className="flex h-6 items-center">
                <input
                  id={option.id}
                  name="hyperparameterTuning"
                  type="radio"
                  checked={selectedTuningMethod === option.id}
                  onChange={() => onTuningMethodSelect(option.id)}
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
        disabled={!selectedTuningMethod}
        className="w-full bg-sky-500 hover:bg-sky-600 disabled:bg-slate-500 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded-lg shadow-md transition duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-opacity-50"
        aria-label="Proceed to Train/Test Split Options"
      >
        Proceed to Train/Test Split Options
      </button>
    </div>
  );
};