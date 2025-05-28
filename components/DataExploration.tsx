
import React, { useState, useCallback } from 'react';
import { CSVRow, SummaryStats, MissingValueInfo, NumericSummary, CategoricalSummary } from '../types';
import { LoadingSpinner } from './LoadingSpinner';
import { CodeDisplay } from './CodeDisplay'; // Re-use for consistency if suitable

interface DataExplorationProps {
  headers: string[];
  sampleData: CSVRow[]; // For quick preview
  fullDataset: CSVRow[]; // For accurate stats - unused directly here, stats are pre-calculated in App.tsx
  summaryStats: SummaryStats | null;
  missingValues: MissingValueInfo[] | null;
  targetVariable: string | null;
  onGenerateCorrelationCode: (headers: string[], targetVariable: string | null) => Promise<string>;
  onProceed: () => void;
  isLoading: boolean; // Pass down main loading state
  setIsLoading: (loading: boolean) => void; // To control loading for correlation code
  setError: (error: string | null) => void;
}

export const DataExploration: React.FC<DataExplorationProps> = ({
  headers,
  sampleData,
  summaryStats,
  missingValues,
  targetVariable,
  onGenerateCorrelationCode,
  onProceed,
  isLoading: isAppLoading, // Rename to avoid conflict
  setIsLoading,
  setError
}) => {
  const [correlationCode, setCorrelationCode] = useState<string | null>(null);
  const [isCorrelationLoading, setIsCorrelationLoading] = useState<boolean>(false);
  const [showCorrelationCode, setShowCorrelationCode] = useState<boolean>(false);
  const [copied, setCopied] = useState(false);


  const handleGenerateCorrelation = async () => {
    if (!process.env.API_KEY) {
        setError("API_KEY environment variable not configured. Cannot generate correlation code.");
        return;
    }
    setIsCorrelationLoading(true);
    setShowCorrelationCode(true); // Show the section immediately
    setCorrelationCode(null); // Clear previous code
    setError(null);
    try {
      const code = await onGenerateCorrelationCode(headers, targetVariable);
      setCorrelationCode(code);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : `Failed to generate correlation code: ${String(err)}`);
      setCorrelationCode("# Error generating code. Please check console or error messages.");
    } finally {
      setIsCorrelationLoading(false);
    }
  };
  
  const handleCopyCode = async (codeToCopy: string | null) => {
    if (!codeToCopy) return;
    try {
      await navigator.clipboard.writeText(codeToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code: ', err);
      alert('Failed to copy code. Please try manually.');
    }
  };

  const renderSummaryStatValue = (value: number | string | null | undefined) => {
    if (value === null || value === undefined) return <span className="text-slate-500 italic">N/A</span>;
    if (typeof value === 'number') return parseFloat(value.toFixed(2)).toLocaleString(); // Format numbers nicely
    return String(value);
  };

  return (
    <div className="space-y-8 p-1">
      {/* Dataset Preview */}
      <section>
        <h3 className="text-xl font-semibold text-sky-400 mb-3">Dataset Preview (First 5 Rows)</h3>
        {sampleData.length > 0 ? (
          <div className="overflow-x-auto bg-slate-700/50 p-4 rounded-lg shadow max-h-80 scrollbar-thin scrollbar-thumb-sky-500 scrollbar-track-slate-600">
            <table className="min-w-full text-sm text-left text-slate-300">
              <thead className="bg-slate-700 sticky top-0">
                <tr>
                  {headers.map((header) => (
                    <th key={header} scope="col" className="px-4 py-2 font-medium whitespace-nowrap">{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-600">
                {sampleData.map((row, rowIndex) => (
                  <tr key={rowIndex} className="hover:bg-slate-600/50 transition-colors">
                    {headers.map((header) => (
                      <td key={`${rowIndex}-${header}`} className="px-4 py-2 whitespace-nowrap">
                        {row[header] !== null && row[header] !== undefined ? String(row[header]) : <span className="text-slate-500 italic">null</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="text-slate-400">No data to preview.</p>}
      </section>

      {/* Summary Statistics */}
      <section>
        <h3 className="text-xl font-semibold text-sky-400 mb-3">Summary Statistics</h3>
        {summaryStats ? (
          <div className="overflow-x-auto bg-slate-700/50 p-4 rounded-lg shadow max-h-96 scrollbar-thin scrollbar-thumb-sky-500 scrollbar-track-slate-600">
            <table className="min-w-full text-sm text-left text-slate-300">
              <thead className="bg-slate-700 sticky top-0">
                <tr>
                  <th className="px-4 py-2 font-medium">Column</th>
                  <th className="px-4 py-2 font-medium">Type</th>
                  <th className="px-4 py-2 font-medium">Count</th>
                  <th className="px-4 py-2 font-medium">Nulls</th>
                  <th className="px-4 py-2 font-medium">Mean/Unique</th>
                  <th className="px-4 py-2 font-medium">Std/Top</th>
                  <th className="px-4 py-2 font-medium">Min/Freq</th>
                  <th className="px-4 py-2 font-medium">Max</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-600">
                {Object.entries(summaryStats).map(([colName, stat]) => (
                  <tr key={colName} className="hover:bg-slate-600/50 transition-colors">
                    <td className="px-4 py-2 font-medium text-sky-300 whitespace-nowrap">{colName}</td>
                    <td className="px-4 py-2 capitalize whitespace-nowrap">{stat.type}</td>
                    <td className="px-4 py-2 whitespace-nowrap">{renderSummaryStatValue(stat.count)}</td>
                    <td className="px-4 py-2 whitespace-nowrap">{renderSummaryStatValue(stat.nulls)}</td>
                    {stat.type === 'numeric' ? (
                      <>
                        <td className="px-4 py-2 whitespace-nowrap">{renderSummaryStatValue((stat as NumericSummary).mean)}</td>
                        <td className="px-4 py-2 whitespace-nowrap">{renderSummaryStatValue((stat as NumericSummary).std)}</td>
                        <td className="px-4 py-2 whitespace-nowrap">{renderSummaryStatValue((stat as NumericSummary).min)}</td>
                        <td className="px-4 py-2 whitespace-nowrap">{renderSummaryStatValue((stat as NumericSummary).max)}</td>
                      </>
                    ) : (
                      <>
                        <td className="px-4 py-2 whitespace-nowrap">{renderSummaryStatValue((stat as CategoricalSummary).unique)}</td>
                        <td className="px-4 py-2 whitespace-nowrap truncate max-w-xs" title={String((stat as CategoricalSummary).top)}>{renderSummaryStatValue((stat as CategoricalSummary).top)}</td>
                        <td className="px-4 py-2 whitespace-nowrap">{renderSummaryStatValue((stat as CategoricalSummary).freq)}</td>
                        <td className="px-4 py-2 text-slate-500 italic">N/A</td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="text-slate-400">Summary statistics are not available.</p>}
      </section>

      {/* Missing Values */}
      <section>
        <h3 className="text-xl font-semibold text-sky-400 mb-3">Missing Values Detection</h3>
        {missingValues && missingValues.length > 0 ? (
          <div className="bg-slate-700/50 p-4 rounded-lg shadow">
            <ul className="space-y-2">
              {missingValues.map((mv) => (
                <li key={mv.column} className="flex justify-between items-center p-2 bg-slate-600/70 rounded">
                  <span className="font-medium text-sky-300">{mv.column}</span>
                  <span className="text-sm text-amber-400">
                    {mv.missingCount} missing ({mv.percentage.toFixed(1)}%)
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ) : <p className="text-slate-400">No missing values detected or data not processed.</p>}
      </section>

      {/* Correlation Matrix Code */}
      <section>
        <h3 className="text-xl font-semibold text-sky-400 mb-3">Correlation Analysis</h3>
        <button
          onClick={handleGenerateCorrelation}
          disabled={isCorrelationLoading || isAppLoading}
          className="w-full sm:w-auto bg-indigo-500 hover:bg-indigo-600 disabled:bg-slate-500 text-white font-semibold py-2 px-4 rounded-lg shadow-md transition duration-150 ease-in-out mb-4"
        >
          {isCorrelationLoading ? 'Generating Code...' : 'Generate Python Code for Correlation Heatmap'}
        </button>
        
        {isCorrelationLoading && <div className="my-4"><LoadingSpinner /></div>}

        {showCorrelationCode && correlationCode && !isCorrelationLoading && (
           <div className="mt-4 animate-fadeIn">
            <CodeDisplay code={correlationCode} />
          </div>
        )}
         {showCorrelationCode && !correlationCode && !isCorrelationLoading && (
            <p className="text-amber-400 mt-2">Click the button above to generate code.</p>
        )}
      </section>

      {/* Proceed Button */}
      <div className="mt-8 pt-6 border-t border-slate-700">
        <button
          onClick={onProceed}
          disabled={isAppLoading || isCorrelationLoading}
          className="w-full bg-sky-500 hover:bg-sky-600 disabled:bg-slate-500 text-white font-semibold py-3 px-4 rounded-lg shadow-md transition duration-150 ease-in-out"
          aria-label="Proceed to advanced preprocessing options"
        >
          Proceed to Advanced Preprocessing
        </button>
      </div>
    </div>
  );
};
