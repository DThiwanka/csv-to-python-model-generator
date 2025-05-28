
import React, { useCallback, useState } from 'react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect }) => {
  const [dragging, setDragging] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      onFileSelect(event.target.files[0]);
    }
  };

  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setDragging(false);
    if (event.dataTransfer.files && event.dataTransfer.files[0]) {
      if (event.dataTransfer.files[0].type === "text/csv" || event.dataTransfer.files[0].name.endsWith('.csv')) {
        onFileSelect(event.dataTransfer.files[0]);
      } else {
        alert("Please upload a valid CSV file.");
      }
    }
  }, [onFileSelect]);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setDragging(false);
  }, []);


  return (
    <div
      className={`p-6 border-2 ${dragging ? 'border-sky-500 bg-slate-700' : 'border-slate-600 border-dashed'} rounded-lg text-center cursor-pointer transition-all duration-200 ease-in-out hover:border-sky-400`}
      onClick={() => document.getElementById('csv-upload')?.click()}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <input
        type="file"
        id="csv-upload"
        accept=".csv"
        onChange={handleFileChange}
        className="hidden"
      />
      <div className="flex flex-col items-center justify-center space-y-3">
        <svg xmlns="http://www.w3.org/2000/svg" className={`h-12 w-12 ${dragging ? 'text-sky-400' : 'text-slate-500'} transition-colors`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
          <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
        <p className={`text-lg font-medium ${dragging ? 'text-sky-300' : 'text-slate-300'}`}>
          {dragging ? "Drop CSV file here" : "Drag & drop a CSV file here, or click to select"}
        </p>
        <p className="text-sm text-slate-400">Max file size: 5MB (for client-side processing)</p>
      </div>
    </div>
  );
};
