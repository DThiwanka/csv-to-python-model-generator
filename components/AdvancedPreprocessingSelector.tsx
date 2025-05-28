
import React, { useState, useEffect } from 'react';
import {
  OutlierMethod, OutlierHandling, ColumnConversionConfig, ForcedColumnType,
  DatetimeExtractionConfig, DatetimeFeature, TextColumnConfig, TextHandlingMethod,
  TargetTransformationMethod
} from '../types';

interface AdvancedPreprocessingSelectorProps {
  headers: string[]; // Feature columns (target already excluded)
  targetVariable: string;
  
  outlierMethod: OutlierMethod;
  onOutlierMethodChange: (method: OutlierMethod) => void;
  outlierHandling: OutlierHandling;
  onOutlierHandlingChange: (handling: OutlierHandling) => void;
  
  columnConversions: ColumnConversionConfig[];
  onColumnConversionsChange: (configs: ColumnConversionConfig[]) => void;
  
  datetimeExtractions: DatetimeExtractionConfig[];
  onDatetimeExtractionsChange: (configs: DatetimeExtractionConfig[]) => void;
  
  textColumnConfigs: TextColumnConfig[];
  onTextColumnConfigsChange: (configs: TextColumnConfig[]) => void;

  targetTransformation: TargetTransformationMethod;
  onTargetTransformationChange: (method: TargetTransformationMethod) => void;

  onProceed: () => void;
}

const forcedTypeOptions: { id: ForcedColumnType; label: string }[] = [
  { id: 'numeric', label: 'Numeric' },
  { id: 'categorical', label: 'Categorical' },
  { id: 'datetime', label: 'Datetime' },
  { id: 'text', label: 'Text' },
];

const datetimeFeatureOptions: { id: DatetimeFeature; label: string }[] = [
  { id: 'year', label: 'Year' },
  { id: 'month', label: 'Month' },
  { id: 'day', label: 'Day' },
  { id: 'weekday', label: 'Weekday' },
  { id: 'hour', label: 'Hour' },
];

const textHandlingOptions: { id: TextHandlingMethod; label: string }[] = [
    { id: 'none', label: 'None (Treat as categorical or drop if not convertible)'},
    { id: 'tfidf', label: 'TF-IDF Vectorizer'},
    { id: 'countvectorizer', label: 'Count Vectorizer'},
];

export const AdvancedPreprocessingSelector: React.FC<AdvancedPreprocessingSelectorProps> = ({
  headers, targetVariable,
  outlierMethod, onOutlierMethodChange,
  outlierHandling, onOutlierHandlingChange,
  columnConversions, onColumnConversionsChange,
  datetimeExtractions, onDatetimeExtractionsChange,
  textColumnConfigs, onTextColumnConfigsChange,
  targetTransformation, onTargetTransformationChange,
  onProceed
}) => {

  const [localColumnConversions, setLocalColumnConversions] = useState<{[key: string]: ForcedColumnType | 'auto'}>({});
  const [localDatetimeExtractions, setLocalDatetimeExtractions] = useState<{[key: string]: DatetimeFeature[]}>({});
  const [localTextConfigs, setLocalTextConfigs] = useState<{[key: string]: TextHandlingMethod}>({});

  useEffect(() => {
    // Initialize local states from props
    const initialConversions: {[key: string]: ForcedColumnType | 'auto'} = {};
    headers.forEach(h => {
      const existing = columnConversions.find(c => c.columnName === h);
      initialConversions[h] = existing ? existing.convertTo : 'auto';
    });
    setLocalColumnConversions(initialConversions);

    const initialDtExtractions: {[key: string]: DatetimeFeature[]} = {};
    datetimeExtractions.forEach(ext => initialDtExtractions[ext.columnName] = ext.features);
    setLocalDatetimeExtractions(initialDtExtractions);

    const initialTextConfigs: {[key: string]: TextHandlingMethod} = {};
    textColumnConfigs.forEach(cfg => initialTextConfigs[cfg.columnName] = cfg.method);
    setLocalTextConfigs(initialTextConfigs);

  }, [headers, columnConversions, datetimeExtractions, textColumnConfigs]);


  const handleLocalConversionChange = (columnName: string, type: ForcedColumnType | 'auto') => {
    setLocalColumnConversions(prev => ({...prev, [columnName]: type}));
     // If changing away from datetime, clear its feature extractions
    if (type !== 'datetime' && localDatetimeExtractions[columnName]) {
      // Pass a valid DatetimeFeature (e.g., the first one) as it's ignored when isClearing is true.
      handleLocalDatetimeFeatureChange(columnName, datetimeFeatureOptions[0].id, true); 
    }
    // If changing away from text, clear its text handling method
    if (type !== 'text' && localTextConfigs[columnName] && localTextConfigs[columnName] !== 'none') {
       handleLocalTextConfigChange(columnName, 'none');
    }
  };
  
  const handleLocalDatetimeFeatureChange = (columnName: string, feature: DatetimeFeature, isClearing: boolean = false) => {
    setLocalDatetimeExtractions(prev => {
      const currentFeatures = prev[columnName] || [];
      let newFeatures;
      if (isClearing) {
        newFeatures = [];
      } else {
        newFeatures = currentFeatures.includes(feature) 
          ? currentFeatures.filter(f => f !== feature)
          : [...currentFeatures, feature];
      }
      return {...prev, [columnName]: newFeatures};
    });
  };

  const handleLocalTextConfigChange = (columnName: string, method: TextHandlingMethod) => {
    setLocalTextConfigs(prev => ({...prev, [columnName]: method}));
  };

  const handleProceedInternal = () => {
    const finalConversions = Object.entries(localColumnConversions)
      .filter(([, type]) => type !== 'auto')
      .map(([columnName, type]) => ({ columnName, convertTo: type as ForcedColumnType }));
    onColumnConversionsChange(finalConversions);

    const finalDtExtractions = Object.entries(localDatetimeExtractions)
        .filter(([, features]) => features && features.length > 0)
        .map(([columnName, features]) => ({columnName, features}));
    onDatetimeExtractionsChange(finalDtExtractions);

    const finalTextConfigs = Object.entries(localTextConfigs)
        .filter(([, method]) => method && method !== 'none')
        .map(([columnName, method]) => ({columnName, method}));
    onTextColumnConfigsChange(finalTextConfigs);
    
    onProceed();
  };

  const getEffectiveColumnType = (columnName: string): ForcedColumnType | 'auto' | undefined => {
    return localColumnConversions[columnName] || 'auto';
  };

  return (
    <div className="space-y-8 p-1">
      {/* Outlier Detection & Handling */}
      <section className="p-4 bg-slate-700/50 rounded-lg shadow">
        <h3 className="text-lg font-semibold text-sky-400 mb-3">1. Outlier Detection & Handling</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Detection Method:</label>
            {['none', 'z_score', 'iqr'].map(method => (
              <div key={method} className="flex items-center mb-1">
                <input type="radio" id={`outlier-${method}`} name="outlierMethod" value={method}
                       checked={outlierMethod === method} onChange={() => onOutlierMethodChange(method as OutlierMethod)}
                       className="h-4 w-4 text-sky-500 border-slate-500 focus:ring-sky-400 bg-slate-600" />
                <label htmlFor={`outlier-${method}`} className="ml-2 text-sm text-slate-200 capitalize">{method.replace('_', ' ')}</label>
              </div>
            ))}
          </div>
          {outlierMethod !== 'none' && (
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">Handling Strategy:</label>
              {['none', 'remove'].map(handling => (
                <div key={handling} className="flex items-center mb-1">
                  <input type="radio" id={`outlier-handle-${handling}`} name="outlierHandling" value={handling}
                         checked={outlierHandling === handling} onChange={() => onOutlierHandlingChange(handling as OutlierHandling)}
                         className="h-4 w-4 text-sky-500 border-slate-500 focus:ring-sky-400 bg-slate-600" />
                  <label htmlFor={`outlier-handle-${handling}`} className="ml-2 text-sm text-slate-200 capitalize">{handling}</label>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* Column Type Conversion */}
      <section className="p-4 bg-slate-700/50 rounded-lg shadow">
        <h3 className="text-lg font-semibold text-sky-400 mb-3">2. Column Type Overrides</h3>
        <p className="text-xs text-slate-400 mb-3">Override auto-detected types if needed. This affects subsequent feature engineering steps.</p>
        <div className="space-y-3 max-h-60 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-700/50 pr-2">
          {headers.map(header => (
            <div key={header} className="grid grid-cols-3 gap-2 items-center">
              <label htmlFor={`type-${header}`} className="text-sm text-slate-200 truncate pr-1" title={header}>{header}:</label>
              <select id={`type-${header}`} value={localColumnConversions[header] || 'auto'}
                      onChange={e => handleLocalConversionChange(header, e.target.value as ForcedColumnType | 'auto')}
                      className="col-span-2 bg-slate-600 border border-slate-500 text-slate-100 text-sm rounded-md p-2 focus:ring-sky-500 focus:border-sky-500">
                <option value="auto">Auto-detect</option>
                {forcedTypeOptions.map(opt => <option key={opt.id} value={opt.id}>{opt.label}</option>)}
              </select>
            </div>
          ))}
        </div>
      </section>

      {/* Datetime Feature Extraction */}
      <section className="p-4 bg-slate-700/50 rounded-lg shadow">
        <h3 className="text-lg font-semibold text-sky-400 mb-3">3. Datetime Feature Extraction</h3>
        <div className="space-y-4 max-h-60 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-700/50 pr-2">
          {headers.filter(h => getEffectiveColumnType(h) === 'datetime').length === 0 && <p className="text-sm text-slate-400">No columns identified or forced as 'datetime'.</p>}
          {headers.filter(h => getEffectiveColumnType(h) === 'datetime').map(header => (
            <div key={`dt-${header}`}>
              <p className="text-sm font-medium text-slate-200 mb-1">{header}:</p>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {datetimeFeatureOptions.map(feat => (
                  <div key={feat.id} className="flex items-center">
                    <input type="checkbox" id={`dt-${header}-${feat.id}`} name={`dt-${header}-${feat.id}`}
                           checked={(localDatetimeExtractions[header] || []).includes(feat.id)}
                           onChange={() => handleLocalDatetimeFeatureChange(header, feat.id)}
                           className="h-4 w-4 text-sky-500 border-slate-500 focus:ring-sky-400 bg-slate-600 rounded" />
                    <label htmlFor={`dt-${header}-${feat.id}`} className="ml-2 text-sm text-slate-300">{feat.label}</label>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Text Feature Handling */}
        <section className="p-4 bg-slate-700/50 rounded-lg shadow">
            <h3 className="text-lg font-semibold text-sky-400 mb-3">4. Text Column Handling</h3>
             <div className="space-y-4 max-h-60 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-700/50 pr-2">
                {headers.filter(h => getEffectiveColumnType(h) === 'text').length === 0 && <p className="text-sm text-slate-400">No columns identified or forced as 'text'.</p>}
                {headers.filter(h => getEffectiveColumnType(h) === 'text').map(header => (
                <div key={`text-${header}`}>
                    <p className="text-sm font-medium text-slate-200 mb-1">{header}:</p>
                    <select value={localTextConfigs[header] || 'none'}
                            onChange={e => handleLocalTextConfigChange(header, e.target.value as TextHandlingMethod)}
                            className="w-full bg-slate-600 border border-slate-500 text-slate-100 text-sm rounded-md p-2 focus:ring-sky-500 focus:border-sky-500">
                    {textHandlingOptions.map(opt => <option key={opt.id} value={opt.id}>{opt.label}</option>)}
                    </select>
                </div>
                ))}
            </div>
        </section>

      {/* Target Variable Transformation */}
      <section className="p-4 bg-slate-700/50 rounded-lg shadow">
        <h3 className="text-lg font-semibold text-sky-400 mb-3">5. Target Variable Transformation ({targetVariable})</h3>
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-1">Transformation Method:</label>
          {[{id: 'none', label: 'None'}, {id: 'log_transform', label: 'Log Transform (for skewed numeric targets, typically in Regression)'}].map(opt => (
            <div key={opt.id} className="flex items-center mb-1">
              <input type="radio" id={`target-transform-${opt.id}`} name="targetTransformation" value={opt.id}
                     checked={targetTransformation === opt.id} 
                     onChange={() => onTargetTransformationChange(opt.id as TargetTransformationMethod)}
                     className="h-4 w-4 text-sky-500 border-slate-500 focus:ring-sky-400 bg-slate-600" />
              <label htmlFor={`target-transform-${opt.id}`} className="ml-2 text-sm text-slate-200">{opt.label}</label>
            </div>
          ))}
        </div>
      </section>

      <div className="mt-8 pt-6 border-t border-slate-700">
        <button
          onClick={handleProceedInternal}
          className="w-full bg-sky-500 hover:bg-sky-600 text-white font-semibold py-3 px-4 rounded-lg shadow-md transition duration-150 ease-in-out"
          aria-label="Proceed to model type selection"
        >
          Proceed to Model Type Selection
        </button>
      </div>
    </div>
  );
};
