
import React, { useState } from 'react';

interface CodeDisplayProps {
  code: string;
}

export const CodeDisplay: React.FC<CodeDisplayProps> = ({ code }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000); // Reset after 2 seconds
    } catch (err) {
      console.error('Failed to copy code: ', err);
      alert('Failed to copy code. Please try manually.');
    }
  };

  return (
    <div className="bg-slate-900 rounded-lg shadow-xl overflow-hidden">
      <div className="bg-slate-700 px-4 py-2 flex justify-between items-center">
        <span className="text-sm font-medium text-slate-300">Python Code</span>
        <button
          onClick={handleCopy}
          className="bg-sky-500 hover:bg-sky-600 text-white text-xs font-semibold py-1 px-3 rounded-md transition-colors duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-sky-400"
        >
          {copied ? (
            <div className="flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              Copied!
            </div>
          ) : (
             <div className="flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                Copy Code
            </div>
          )}
        </button>
      </div>
      <pre className="p-4 sm:p-6 text-sm overflow-x-auto bg-slate-800 scrollbar-thin scrollbar-thumb-sky-400 scrollbar-track-slate-700 hover:scrollbar-thumb-sky-500">
        <code className="language-python text-slate-200 whitespace-pre-wrap break-words">
          {code}
        </code>
      </pre>
    </div>
  );
};
