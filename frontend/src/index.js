import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css'; // Tailwind CSS나 기본 스타일을 불러옵니다.
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);