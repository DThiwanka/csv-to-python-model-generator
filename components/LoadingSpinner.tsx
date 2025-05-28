
import React from 'react';

export const LoadingSpinner: React.FC = () => {
  return (
    <div className="flex items-center justify-center space-x-2">
      <div className="w-5 h-5 rounded-full animate-pulse bg-sky-400"></div>
      <div className="w-5 h-5 rounded-full animate-pulse bg-sky-400" style={{ animationDelay: '0.2s' }}></div>
      <div className="w-5 h-5 rounded-full animate-pulse bg-sky-400" style={{ animationDelay: '0.4s' }}></div>
      <span className="ml-3 text-slate-300">Loading...</span>
    </div>
  );
};
