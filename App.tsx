
import React, { useState, useCallback, useEffect } from 'react';
import { GoogleGenAI } from '@google/genai';
import { FileUpload } from './components/FileUpload';
import { ColumnSelector } from './components/ColumnSelector';
import { DataExploration } from './components/DataExploration';
import { AdvancedPreprocessingSelector } from './components/AdvancedPreprocessingSelector'; // New
import { ModelTypeSelector } from './components/ModelTypeSelector';
import { AlgorithmSelector } from './components/AlgorithmSelector';
import { HyperparameterTuningSelector } from './components/HyperparameterTuningSelector';
import { TrainTestSplitSelector } from './components/TrainTestSplitSelector';
import { ModelPersistenceSelector } from './components/ModelPersistenceSelector';
import { EvaluationSelector } from './components/EvaluationSelector';
import { CodeDisplay } from './components/CodeDisplay';
import { LoadingSpinner } from './components/LoadingSpinner';
import { ErrorMessage } from './components/ErrorMessage';
import { suggestTargetVariable, generatePythonModelCode, generateCorrelationMatrixCode } from './services/geminiService';
import { 
  AppStep, CSVRow, ModelType, ModelPersistenceOption, 
  SummaryStats, MissingValueInfo, Algorithm, HyperparameterTuningMethod,
  ColumnSummary, NumericSummary, CategoricalSummary,
  ClassificationMetric, RegressionMetric, EvaluationMetric,
  ClassificationVisualization, RegressionVisualization, EvaluationVisualization,
  TrainTestSplitRatio, StratifyOption,
  // New Advanced Preprocessing Types
  OutlierMethod, OutlierHandling, ColumnConversionConfig,
  DatetimeExtractionConfig, TextColumnConfig, TargetTransformationMethod
} from './types';

// Ensure PapaParse is available globally for TypeScript
declare var Papa: any;

export const App: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<AppStep>('initial');
  const [file, setFile] = useState<File | null>(null);
  const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
  const [csvSample, setCsvSample] = useState<CSVRow[]>([]);
  const [fullCsvData, setFullCsvData] = useState<CSVRow[]>([]);
  const [suggestedTarget, setSuggestedTarget] = useState<string | null>(null);
  const [selectedTarget, setSelectedTarget] = useState<string | null>(null);
  
  // Advanced Preprocessing State
  const [selectedOutlierMethod, setSelectedOutlierMethod] = useState<OutlierMethod>('none');
  const [selectedOutlierHandling, setSelectedOutlierHandling] = useState<OutlierHandling>('none');
  const [columnConversions, setColumnConversions] = useState<ColumnConversionConfig[]>([]);
  const [datetimeExtractions, setDatetimeExtractions] = useState<DatetimeExtractionConfig[]>([]);
  const [textColumnConfigs, setTextColumnConfigs] = useState<TextColumnConfig[]>([]);
  const [selectedTargetTransformation, setSelectedTargetTransformation] = useState<TargetTransformationMethod>('none');

  const [selectedModelType, setSelectedModelType] = useState<ModelType | null>(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<Algorithm | null>(null);
  const [selectedHyperparameterTuning, setSelectedHyperparameterTuning] = useState<HyperparameterTuningMethod | null>(null);
  
  const [selectedTrainTestSplitRatio, setSelectedTrainTestSplitRatio] = useState<TrainTestSplitRatio>('80/20');
  const [selectedStratifyOption, setSelectedStratifyOption] = useState<StratifyOption>('yes');

  const [selectedModelPersistence, setSelectedModelPersistence] = useState<ModelPersistenceOption | null>(null);
  
  const [selectedClassificationMetrics, setSelectedClassificationMetrics] = useState<ClassificationMetric[]>([]);
  const [selectedRegressionMetrics, setSelectedRegressionMetrics] = useState<RegressionMetric[]>([]);
  const [selectedClassificationVisualizations, setSelectedClassificationVisualizations] = useState<ClassificationVisualization[]>([]);
  const [selectedRegressionVisualizations, setSelectedRegressionVisualizations] = useState<RegressionVisualization[]>([]);

  const [pythonCode, setPythonCode] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const [summaryStatistics, setSummaryStatistics] = useState<SummaryStats | null>(null);
  const [missingValueInfo, setMissingValueInfo] = useState<MissingValueInfo[] | null>(null);


  const resetState = () => {
    setCurrentStep('initial');
    setFile(null);
    setCsvHeaders([]);
    setCsvSample([]);
    setFullCsvData([]);
    setSuggestedTarget(null);
    setSelectedTarget(null);
    
    setSelectedOutlierMethod('none');
    setSelectedOutlierHandling('none');
    setColumnConversions([]);
    setDatetimeExtractions([]);
    setTextColumnConfigs([]);
    setSelectedTargetTransformation('none');

    setSelectedModelType(null);
    setSelectedAlgorithm(null);
    setSelectedHyperparameterTuning(null);
    setSelectedTrainTestSplitRatio('80/20'); 
    setSelectedStratifyOption('yes'); 
    setSelectedModelPersistence(null);
    setSelectedClassificationMetrics([]);
    setSelectedRegressionMetrics([]);
    setSelectedClassificationVisualizations([]);
    setSelectedRegressionVisualizations([]);
    setPythonCode(null);
    setError(null);
    setIsLoading(false);
    setSummaryStatistics(null);
    setMissingValueInfo(null);
  };
  
  const calculateSummaryStatistics = (data: CSVRow[], headers: string[]): SummaryStats => {
    const stats: SummaryStats = {};
    headers.forEach(header => {
      const values = data.map(row => row[header]).filter(val => val !== null && val !== undefined && val !== '');
      const numericValues = values.map(v => Number(v)).filter(n => !isNaN(n));
      const nullCount = data.length - values.length;

      if (numericValues.length / values.length > 0.7) { 
        const sum = numericValues.reduce((acc, val) => acc + val, 0);
        const mean = numericValues.length > 0 ? sum / numericValues.length : null;
        const sorted = numericValues.slice().sort((a, b) => a - b);
        const min = sorted.length > 0 ? sorted[0] : null;
        const max = sorted.length > 0 ? sorted[sorted.length - 1] : null;
        const std = numericValues.length > 1 
          ? Math.sqrt(numericValues.map(x => Math.pow(x - (mean!), 2)).reduce((a, b) => a + b, 0) / (numericValues.length -1))
          : null;
        
        stats[header] = { type: 'numeric', count: numericValues.length, mean, std, min, max, nulls: nullCount } as NumericSummary;
      } else { 
        const valueCounts: { [key: string]: number } = {};
        values.forEach(val => {
          const valStr = String(val);
          valueCounts[valStr] = (valueCounts[valStr] || 0) + 1;
        });
        const uniqueCount = Object.keys(valueCounts).length;
        let top: string | number | null = null;
        let freq: number | null = 0;
        if (uniqueCount > 0) {
          const sortedCounts = Object.entries(valueCounts).sort(([,a],[,b]) => b-a);
          top = sortedCounts[0][0];
          freq = sortedCounts[0][1];
        }
        stats[header] = { type: 'categorical', count: values.length, unique: uniqueCount, top, freq, nulls: nullCount } as CategoricalSummary;
      }
    });
    return stats;
  };

  const calculateMissingValueInfo = (data: CSVRow[], headers: string[]): MissingValueInfo[] => {
    return headers.map(header => {
      const missingCount = data.filter(row => row[header] === null || row[header] === undefined || row[header] === '').length;
      return { column: header, missingCount, totalCount: data.length, percentage: data.length > 0 ? (missingCount / data.length) * 100 : 0, };
    }).filter(info => info.missingCount > 0);
  };

  const handleFileSelect = useCallback(async (selectedFile: File) => {
    if (!process.env.API_KEY) {
      setError("API_KEY environment variable not configured.");
      setCurrentStep('error');
      return;
    }
    setFile(selectedFile);
    setCurrentStep('file_processing');
    setIsLoading(true);
    setError(null);

    Papa.parse(selectedFile, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: async (results: any) => {
        if (results.errors.length > 0) {
          setError(`Error parsing CSV: ${results.errors[0].message}`);
          setCurrentStep('error');
          setIsLoading(false);
          return;
        }
        if (!results.meta.fields || results.meta.fields.length === 0) {
          setError("CSV has no headers or is empty.");
          setCurrentStep('error');
          setIsLoading(false);
          return;
        }
        
        const headers = results.meta.fields as string[];
        const allData = results.data as CSVRow[];
        
        setCsvHeaders(headers);
        setFullCsvData(allData);
        setCsvSample(allData.slice(0, 5));

        try {
          const target = await suggestTargetVariable(headers);
          setSuggestedTarget(target);
          setSelectedTarget(target); 
          setCurrentStep('target_selection');
        } catch (err) {
          console.error(err);
          setError(err instanceof Error ? err.message : String(err));
          setCurrentStep('error');
        } finally {
          setIsLoading(false);
        }
      },
      error: (err: Error) => {
        setError(`File parsing error: ${err.message}`);
        setCurrentStep('error');
        setIsLoading(false);
      }
    });
  }, []);

  const handleTargetSelection = useCallback((target: string) => {
    setSelectedTarget(target);
  }, []);

  const handleProceedToDataExploration = useCallback(() => {
    if (!selectedTarget) {
      setError("Please select a target variable first.");
      setCurrentStep('error'); 
      return;
    }
    if (fullCsvData.length > 0 && csvHeaders.length > 0) {
      setSummaryStatistics(calculateSummaryStatistics(fullCsvData, csvHeaders));
      setMissingValueInfo(calculateMissingValueInfo(fullCsvData, csvHeaders));
    }
    setCurrentStep('data_exploration');
  }, [selectedTarget, fullCsvData, csvHeaders]);

  const handleProceedToAdvancedPreprocessing = useCallback(() => { // New handler
    setCurrentStep('advanced_preprocessing_selection');
  }, []);
  
  const handleProceedToModelTypeFromAdvanced = useCallback(() => { // New handler
    setCurrentStep('model_type_selection');
  }, []);

  const handleModelTypeSelect = useCallback((modelType: ModelType) => {
    setSelectedModelType(modelType);
    setSelectedAlgorithm(null); 
    setSelectedHyperparameterTuning(null);
    setSelectedClassificationMetrics([]);
    setSelectedRegressionMetrics([]);
    setSelectedClassificationVisualizations([]);
    setSelectedRegressionVisualizations([]);
    if (modelType === 'regression') {
      setSelectedStratifyOption('no'); 
    } else {
      setSelectedStratifyOption('yes');
    }
  }, []);

  const handleProceedToAlgorithmSelection = useCallback(() => {
    if (!selectedModelType) {
      setError("Please select a model type first.");
      setCurrentStep('error');
      return;
    }
    setCurrentStep('algorithm_selection');
  }, [selectedModelType]);

  const handleAlgorithmSelect = useCallback((algorithm: Algorithm) => {
    setSelectedAlgorithm(algorithm);
    setSelectedHyperparameterTuning(null); 
  }, []);

  const handleProceedToHyperparameterTuning = useCallback(() => {
    if (!selectedAlgorithm) {
      setError("Please select an algorithm first.");
      setCurrentStep('error');
      return;
    }
    setCurrentStep('hyperparameter_tuning_selection');
  }, [selectedAlgorithm]);

  const handleHyperparameterTuningSelect = useCallback((method: HyperparameterTuningMethod) => {
    setSelectedHyperparameterTuning(method);
  }, []);

  const handleProceedToTrainTestSplitOptions = useCallback(() => {
    if(!selectedModelType || !selectedAlgorithm) {
      setError("Model type or algorithm not selected.");
      setCurrentStep('error');
      return;
    }
    if (!selectedHyperparameterTuning) {
        setError("Please select a hyperparameter tuning option.");
        setCurrentStep('error');
        return;
    }
    setCurrentStep('train_test_split_options');
  }, [selectedModelType, selectedAlgorithm, selectedHyperparameterTuning]);

  const handleTrainTestSplitRatioSelect = useCallback((ratio: TrainTestSplitRatio) => {
    setSelectedTrainTestSplitRatio(ratio);
  }, []);

  const handleStratifyOptionSelect = useCallback((option: StratifyOption) => {
    setSelectedStratifyOption(option);
  }, []);

  const handleProceedToModelPersistence = useCallback(() => {
    setCurrentStep('model_persistence_selection');
  }, []);

  const handleModelPersistenceSelect = useCallback((persistenceOption: ModelPersistenceOption) => {
    setSelectedModelPersistence(persistenceOption);
  }, []);

  const handleProceedToEvaluationSelection = useCallback(() => {
    if (!selectedModelPersistence) {
      setError("Please select a model persistence option.");
      setCurrentStep('error');
      return;
    }
    setCurrentStep('evaluation_selection');
  }, [selectedModelPersistence]);

  const handleClassificationMetricsChange = useCallback((metrics: ClassificationMetric[]) => {
    setSelectedClassificationMetrics(metrics);
  }, []);
  const handleRegressionMetricsChange = useCallback((metrics: RegressionMetric[]) => {
    setSelectedRegressionMetrics(metrics);
  }, []);
  const handleClassificationVisualizationsChange = useCallback((visualizations: ClassificationVisualization[]) => {
    setSelectedClassificationVisualizations(visualizations);
  }, []);
  const handleRegressionVisualizationsChange = useCallback((visualizations: RegressionVisualization[]) => {
    setSelectedRegressionVisualizations(visualizations);
  }, []);


  const handleGenerateCode = useCallback(async () => {
    if (!selectedTarget || csvHeaders.length === 0 || !selectedModelType || 
        !selectedAlgorithm || !selectedHyperparameterTuning || 
        !selectedTrainTestSplitRatio || 
        !selectedModelPersistence) {
      setError("One or more primary selections are missing. Please complete all steps.");
      setCurrentStep('error');
      return;
    }
    if (selectedModelType === 'classification' && (selectedClassificationMetrics.length === 0 && selectedClassificationVisualizations.length === 0)) {
        setError("Please select at least one metric or visualization for classification model evaluation.");
        setCurrentStep('error'); // Stay on current step or guide user
        return;
    }
    if (selectedModelType === 'regression' && (selectedRegressionMetrics.length === 0 && selectedRegressionVisualizations.length === 0)) {
        setError("Please select at least one metric or visualization for regression model evaluation.");
        setCurrentStep('error'); // Stay on current step or guide user
        return;
    }

    if (!process.env.API_KEY) {
        setError("API_KEY environment variable not configured. Cannot generate code.");
        setCurrentStep('error');
        return;
    }

    setCurrentStep('code_generation');
    setIsLoading(true);
    setError(null);

    const metrics = selectedModelType === 'classification' ? selectedClassificationMetrics : selectedRegressionMetrics;
    const visualizations = selectedModelType === 'classification' ? selectedClassificationVisualizations : selectedRegressionVisualizations;

    try {
      const code = await generatePythonModelCode(
        csvHeaders, 
        selectedTarget, 
        csvSample, // Sample data remains useful for context
        // Advanced Preprocessing Options
        selectedOutlierMethod,
        selectedOutlierHandling,
        columnConversions,
        datetimeExtractions,
        textColumnConfigs,
        selectedTargetTransformation,
        // Model Configuration Options
        selectedModelType,
        selectedAlgorithm, 
        selectedHyperparameterTuning,
        selectedTrainTestSplitRatio, 
        selectedStratifyOption,      
        selectedModelPersistence,
        metrics as EvaluationMetric[],
        visualizations as EvaluationVisualization[]
      );
      setPythonCode(code);
      setCurrentStep('code_display');
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : String(err));
      setCurrentStep('error');
    } finally {
      setIsLoading(false);
    }
  }, [
      selectedTarget, csvHeaders, csvSample, 
      selectedOutlierMethod, selectedOutlierHandling, columnConversions, datetimeExtractions, textColumnConfigs, selectedTargetTransformation, // New advanced options
      selectedModelType, selectedAlgorithm, 
      selectedHyperparameterTuning, selectedTrainTestSplitRatio, selectedStratifyOption, 
      selectedModelPersistence, selectedClassificationMetrics, selectedRegressionMetrics,
      selectedClassificationVisualizations, selectedRegressionVisualizations
  ]);
  
  useEffect(() => {
    const apiKeyErrorMsg1 = "API_KEY environment variable not configured. Please ensure it is set up correctly in your environment.";
    const apiKeyErrorMsg2 = "API_KEY environment variable not configured.";
    if (process.env.API_KEY && (error === apiKeyErrorMsg1 || error === apiKeyErrorMsg2)) {
      setError(null);
      if(currentStep === 'error') setCurrentStep('initial');
    }
  }, [error, currentStep]);


  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-slate-100 flex flex-col items-center p-4 sm:p-8">
      <header className="w-full max-w-4xl mb-8 text-center">
        <h1 className="text-4xl sm:text-5xl font-bold text-sky-400">CSV to Python Model Generator</h1>
        <p className="text-slate-400 mt-2 text-lg">Upload CSV, explore, preprocess, configure options, get AI-generated Python code.</p>
      </header>

      <main className="w-full max-w-4xl bg-slate-800 shadow-2xl rounded-xl p-6 sm:p-10">
        {isLoading && currentStep !== 'data_exploration' && currentStep !== 'advanced_preprocessing_selection' && <div className="flex justify-center my-6"><LoadingSpinner /></div>}
        
        {error && !isLoading && (
          <div className="my-6">
            <ErrorMessage message={error} />
            <button
              onClick={resetState}
              className="mt-4 w-full bg-sky-500 hover:bg-sky-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md transition duration-150 ease-in-out"
            >
              Start Over
            </button>
          </div>
        )}

        {!isLoading && !error && (
          <>
            {currentStep === 'initial' && (
               <div className="animate-fadeIn">
                <h2 className="text-2xl font-semibold text-sky-400 mb-4">1. Upload CSV File</h2>
                <FileUpload onFileSelect={handleFileSelect} />
              </div>
            )}

            {currentStep === 'file_processing' && <div className="flex justify-center my-6"><LoadingSpinner /><p className="ml-3 text-slate-300">Processing your file...</p></div> }


            {currentStep === 'target_selection' && csvHeaders.length > 0 && (
              <div className="animate-fadeIn">
                <h2 className="text-2xl font-semibold text-sky-400 mb-4">2. Select Target Variable</h2>
                <ColumnSelector
                  headers={csvHeaders}
                  suggestedTarget={suggestedTarget}
                  selectedTarget={selectedTarget}
                  onTargetSelect={handleTargetSelection}
                  onProceed={handleProceedToDataExploration}
                />
              </div>
            )}
            
            {currentStep === 'data_exploration' && (
              <div className="animate-fadeIn">
                <h2 className="text-2xl font-semibold text-sky-400 mb-4">3. Data Exploration</h2>
                 <DataExploration
                  headers={csvHeaders}
                  sampleData={csvSample}
                  fullDataset={fullCsvData}
                  summaryStats={summaryStatistics}
                  missingValues={missingValueInfo}
                  targetVariable={selectedTarget}
                  onGenerateCorrelationCode={generateCorrelationMatrixCode}
                  onProceed={handleProceedToAdvancedPreprocessing} // Changed to new step
                  isLoading={isLoading}
                  setIsLoading={setIsLoading}
                  setError={setError}
                />
              </div>
            )}
            
            {currentStep === 'advanced_preprocessing_selection' && selectedTarget && ( // New Step
              <div className="animate-fadeIn">
                <h2 className="text-2xl font-semibold text-sky-400 mb-4">4. Advanced Preprocessing Options</h2>
                <AdvancedPreprocessingSelector
                  headers={csvHeaders.filter(h => h !== selectedTarget)} // Pass only feature columns
                  targetVariable={selectedTarget}
                  outlierMethod={selectedOutlierMethod}
                  onOutlierMethodChange={setSelectedOutlierMethod}
                  outlierHandling={selectedOutlierHandling}
                  onOutlierHandlingChange={setSelectedOutlierHandling}
                  columnConversions={columnConversions}
                  onColumnConversionsChange={setColumnConversions}
                  datetimeExtractions={datetimeExtractions}
                  onDatetimeExtractionsChange={setDatetimeExtractions}
                  textColumnConfigs={textColumnConfigs}
                  onTextColumnConfigsChange={setTextColumnConfigs}
                  targetTransformation={selectedTargetTransformation}
                  onTargetTransformationChange={setSelectedTargetTransformation}
                  onProceed={handleProceedToModelTypeFromAdvanced}
                />
              </div>
            )}


            {currentStep === 'model_type_selection' && (
              <div className="animate-fadeIn">
                <h2 className="text-2xl font-semibold text-sky-400 mb-4">5. Select Model Type</h2>
                <ModelTypeSelector
                  selectedModelType={selectedModelType}
                  onModelTypeSelect={handleModelTypeSelect}
                  onProceed={handleProceedToAlgorithmSelection}
                />
              </div>
            )}

            {currentStep === 'algorithm_selection' && selectedModelType && (
              <div className="animate-fadeIn">
                <h2 className="text-2xl font-semibold text-sky-400 mb-4">6. Select Specific Algorithm</h2>
                <AlgorithmSelector
                  modelType={selectedModelType}
                  selectedAlgorithm={selectedAlgorithm}
                  onAlgorithmSelect={handleAlgorithmSelect}
                  onProceed={handleProceedToHyperparameterTuning}
                />
              </div>
            )}

            {currentStep === 'hyperparameter_tuning_selection' && selectedAlgorithm && (
              <div className="animate-fadeIn">
                <h2 className="text-2xl font-semibold text-sky-400 mb-4">7. Hyperparameter Tuning</h2>
                <HyperparameterTuningSelector
                  selectedAlgorithm={selectedAlgorithm} 
                  selectedTuningMethod={selectedHyperparameterTuning}
                  onTuningMethodSelect={handleHyperparameterTuningSelect}
                  onProceed={handleProceedToTrainTestSplitOptions}
                />
              </div>
            )}

            {currentStep === 'train_test_split_options' && selectedModelType && (
              <div className="animate-fadeIn">
                <h2 className="text-2xl font-semibold text-sky-400 mb-4">8. Train/Test Split Options</h2>
                <TrainTestSplitSelector
                  modelType={selectedModelType}
                  selectedRatio={selectedTrainTestSplitRatio}
                  onRatioChange={handleTrainTestSplitRatioSelect}
                  selectedStratify={selectedStratifyOption}
                  onStratifyChange={handleStratifyOptionSelect}
                  onProceed={handleProceedToModelPersistence}
                />
              </div>
            )}

            {currentStep === 'model_persistence_selection' && (
              <div className="animate-fadeIn">
                <h2 className="text-2xl font-semibold text-sky-400 mb-4">9. Model Persistence</h2>
                <ModelPersistenceSelector
                  selectedOption={selectedModelPersistence}
                  onOptionSelect={handleModelPersistenceSelect}
                  onProceed={handleProceedToEvaluationSelection}
                />
              </div>
            )}

            {currentStep === 'evaluation_selection' && selectedModelType && (
              <div className="animate-fadeIn">
                <h2 className="text-2xl font-semibold text-sky-400 mb-4">10. Select Evaluation Options</h2>
                <EvaluationSelector
                  modelType={selectedModelType}
                  selectedClassificationMetrics={selectedClassificationMetrics}
                  onClassificationMetricsChange={handleClassificationMetricsChange}
                  selectedRegressionMetrics={selectedRegressionMetrics}
                  onRegressionMetricsChange={handleRegressionMetricsChange}
                  selectedClassificationVisualizations={selectedClassificationVisualizations}
                  onClassificationVisualizationsChange={handleClassificationVisualizationsChange}
                  selectedRegressionVisualizations={selectedRegressionVisualizations}
                  onRegressionVisualizationsChange={handleRegressionVisualizationsChange}
                  onSubmit={handleGenerateCode}
                />
              </div>
            )}
            
            {currentStep === 'code_generation' && <div className="flex justify-center my-6"><LoadingSpinner /><p className="ml-3 text-slate-300">Generating Python code...</p></div> }


            {currentStep === 'code_display' && pythonCode && (
              <div className="animate-fadeIn">
                <h2 className="text-2xl font-semibold text-sky-400 mb-4">11. Generated Python Code</h2>
                <CodeDisplay code={pythonCode} />
                 <button
                    onClick={resetState}
                    className="mt-6 w-full bg-sky-500 hover:bg-sky-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md transition duration-150 ease-in-out"
                  >
                    Analyze Another CSV
                  </button>
              </div>
            )}
          </>
        )}
      </main>
      <footer className="w-full max-w-4xl mt-12 text-center">
        <p className="text-slate-500 text-sm">Powered by Google Gemini & React</p>
      </footer>
    </div>
  );
};
