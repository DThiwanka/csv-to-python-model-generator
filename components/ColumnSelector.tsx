import React from 'react';

interface ColumnSelectorProps {
  headers: string[];
  suggestedTarget: string | null;
  selectedTarget: string | null;
  onTargetSelect: (column: string) => void;
  onProceed: () => void;
}

export const ColumnSelector: React.FC<ColumnSelectorProps> = ({
  headers,
  suggestedTarget,
  selectedTarget,
  onTargetSelect,
  onProceed,
}) => {
  return (
    <div className="space-y-6 p-6 bg-slate-700 rounded-lg shadow-lg">
      <div>
        <h3 className="text-xl font-semibold text-sky-400 mb-1">Detected Headers:</h3>
        <div className="flex flex-wrap gap-2 mb-4">
          {headers.map((header) => (
            <span key={header} className="bg-slate-600 text-slate-200 px-3 py-1 rounded-full text-sm shadow">
              {header}
            </span>
          ))}
        </div>
      </div>
      
      {suggestedTarget && (
        <p className="text-slate-300">
          AI Suggestion for target variable: <strong className="text-sky-400">{suggestedTarget}</strong>
        </p>
      )}

      <div className="space-y-3">
        <label htmlFor="target-column" className="block text-lg font-medium text-slate-200">
          Choose the column to predict (target variable):
        </label>
        <select
          id="target-column"
          value={selectedTarget || ''}
          onChange={(e) => onTargetSelect(e.target.value)}
          className="w-full p-3 bg-slate-600 border border-slate-500 text-slate-100 rounded-md shadow-sm focus:ring-sky-500 focus:border-sky-500 transition-colors"
          aria-label="Select target variable"
        >
          <option value="" disabled>
            Select a column
          </option>
          {headers.map((header) => (
            <option key={header} value={header}>
              {header}
            </option>
          ))}
        </select>
      </div>

      <button
        onClick={onProceed}
        disabled={!selectedTarget}
        className="w-full bg-sky-500 hover:bg-sky-600 disabled:bg-slate-500 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded-lg shadow-md transition duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-opacity-50"
      >
        Proceed to Data Exploration
      </button>
    </div>
  );
};