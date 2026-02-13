import React, { useState, useRef, useEffect } from 'react';
import { 
  PaperAirplaneIcon, PaperClipIcon, XMarkIcon, DocumentIcon, 
  ChatBubbleLeftRightIcon, Bars3Icon, PlusIcon, ChevronLeftIcon 
} from '@heroicons/react/24/outline';
import logo from './logo.png';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [messages, setMessages] = useState([{ id: 1, text: "Hello! How can I help you?", isUser: false }]);
  const [chatHistory, setChatHistory] = useState([]);
  const [inputText, setInputText] = useState("");
  
  // 1. 상태 변수명을 selectedFiles(배열)로 통일
  const [selectedFiles, setSelectedFiles] = useState([]); 
  const [isLoading, setIsLoading] = useState(false);
  
  const scrollRef = useRef(null);
  const fileInputRef = useRef(null);

  const fetchHistory = () => {
    fetch('http://localhost:8000/history')
      .then(res => res.json())
      .then(data => setChatHistory(data.history || []))
      .catch(err => console.error("History fetch error"));
  };

  useEffect(() => { fetchHistory(); }, []);
  useEffect(() => { scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight); }, [messages]);

  const handleNewChat = () => {
    setMessages([{ id: 1, text: "Starting a new conversation. Please enter your question.", isUser: false }]);
    setSelectedFiles([]);
    setInputText("");
  };

  const handleHistoryClick = async (queryText) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append('prompt', queryText);
    try {
      const response = await fetch('http://localhost:8000/chat', { method: 'POST', body: formData });
      const data = await response.json();
      setMessages([
        { id: Date.now(), text: queryText, isUser: true },
        { id: Date.now() + 1, text: data.answer, isUser: false }
      ]);
    } catch (error) { console.error("Error"); } finally { setIsLoading(false); }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputText.trim() && selectedFiles.length === 0) return; // 조건 수정

    // 사용자 메시지에 표시할 모든 파일 이름 결합
    const fileNames = selectedFiles.map(f => f.name).join(", ");
    const userMessage = { id: Date.now(), text: inputText, fileName: fileNames, isUser: true };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    const formData = new FormData();
    formData.append('prompt', inputText);

    // 2. 모든 파일을 'files' 키로 추가
    selectedFiles.forEach(file => {
      formData.append('files', file); 
    });

    try {
      const response = await fetch('http://localhost:8000/chat', { method: 'POST', body: formData });
      const data = await response.json();
      setMessages(prev => [...prev, { id: Date.now() + 1, text: data.answer, isUser: false }]);
      fetchHistory();
    } catch (error) { 
      setMessages(prev => [...prev, { id: Date.now() + 1, text: "서버 통신 오류", isUser: false }]);
    } finally { 
      setIsLoading(false); 
      setInputText(""); 
      setSelectedFiles([]); // 전송 후 비우기
    }
  };

  return (
    <div className="flex h-screen bg-[#0a0a0a] text-white font-sans overflow-hidden">
      
      {/* --- 왼쪽 사이드바 (다크 톤) --- */}
      <aside className={`bg-[#111111] border-r border-white/5 flex flex-col transition-all duration-300 ${isSidebarOpen ? 'w-72' : 'w-0 opacity-0 pointer-events-none'}`}>
        <div className="p-4 border-b border-white/5">
          <button 
            onClick={handleNewChat}
            className="w-full flex items-center justify-center gap-2 p-3 bg-white/5 border border-white/10 rounded-xl hover:bg-white/10 transition-all font-bold text-sm text-white"
          >
            <PlusIcon className="h-5 w-5" /> New Chat
          </button>
        </div>
        
        <div className="p-4 font-bold text-[10px] text-gray-500 tracking-[0.2em] uppercase">History</div>
        <nav className="flex-1 overflow-y-auto px-2 space-y-1">
          {chatHistory.map((item, idx) => (
            <div 
              key={idx} 
              onClick={() => handleHistoryClick(item)}
              className="flex items-center gap-3 p-3 text-sm hover:bg-white/5 rounded-xl cursor-pointer transition-all group"
            >
              <ChatBubbleLeftRightIcon className="h-4 w-4 text-gray-500 group-hover:text-white" />
              <span className="truncate text-gray-400 group-hover:text-white">{item}</span>
            </div>
          ))}
        </nav>

        <button 
          onClick={() => setIsSidebarOpen(false)}
          className="p-4 flex items-center justify-center text-gray-500 hover:text-white transition-all border-t border-white/5"
        >
          <ChevronLeftIcon className="h-5 w-5 mr-2" /> Close Sidebar
        </button>
      </aside>

      {/* --- 메인 채팅창 (요청하신 검정 배경) --- */}
      <div className="flex-1 flex flex-col relative bg-[#0a0a0a]">
        
        {!isSidebarOpen && (
          <button 
            onClick={() => setIsSidebarOpen(true)}
            className="absolute top-6 left-6 z-20 p-2 bg-white text-black rounded-full shadow-lg hover:scale-110 transition-all"
          >
            <Bars3Icon className="h-5 w-5" />
          </button>
        )}

        <header className="bg-[#0a0a0a] py-4 px-8 flex items-center justify-between border-b border-white/5 z-10">
          <div className={`flex items-center gap-4 transition-all duration-300 ${!isSidebarOpen ? 'ml-12' : 'ml-0'}`}>
            <img 
              src={logo} 
              alt="Ctrl Logo" 
              className="w-10 h-10 object-contain drop-shadow-[0_0_8px_rgba(255,255,255,0.2)] invert" 
            /> 
            <div className="flex flex-col border-l border-white/20 pl-4">
              <span className="text-white font-black text-3xl tracking-tighter leading-none">Ctrl</span>
            </div>
          </div>
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.4)]"></div>
        </header>

        {/* 화면 중앙 은은한 배경 로고 (에러 수정됨) */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-0 overflow-hidden">
          <img 
            src={logo} 
            alt="Watermark" 
            className="w-[450px] h-[450px] object-contain opacity-[0.08] grayscale invert" 
          />
        </div>

        <main ref={scrollRef} className="flex-1 overflow-y-auto p-8 space-y-8 max-w-5xl mx-auto w-full scroll-smooth z-10 relative">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'} animate-in fade-in duration-500`}>
              <div className={`max-w-[80%] rounded-3xl px-6 py-4 shadow-sm ${
                msg.isUser 
                  ? 'bg-white text-black font-semibold' 
                  : 'bg-[#1a1a1a] text-gray-200 border border-white/5'
              }`}>
                {msg.fileName && (
                  <div className="flex items-center gap-2 mb-2 p-2 bg-white/5 rounded-lg text-[10px] text-gray-400 italic">
                    <DocumentIcon className="h-4 w-4" /> {msg.fileName}
                  </div>
                )}
                <div className="prose prose-invert max-w-none text-sm md:text-base leading-relaxed">
                  <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
                    {msg.text}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex items-center gap-2 text-gray-600 text-xs animate-pulse font-mono px-6">
              <div className="w-2 h-2 bg-white rounded-full animate-bounce" />
              Processing Database...
            </div>
          )}
        </main>

        <footer className="p-8 bg-[#0a0a0a] border-t border-white/5 z-10">
          <div className="max-w-5xl mx-auto">
           {/* 3. 선택된 파일들을 리스트로 보여주는 UI 수정 */}
            <div className="flex flex-wrap gap-2 mb-3">
              {selectedFiles.map((file, index) => (
                <div key={index} className="flex items-center bg-white text-black p-2 px-4 rounded-full text-xs shadow-lg font-bold">
                  <DocumentIcon className="h-4 w-4 mr-2" />
                  <span className="truncate max-w-[150px]">{file.name}</span>
                  <button 
                    type="button"
                    onClick={() => setSelectedFiles(prev => prev.filter((_, i) => i !== index))} 
                    className="ml-2 hover:text-red-600"
                  >
                    <XMarkIcon className="h-4 w-4" />
                  </button>
                </div>
              ))}
            </div>
            
            <form onSubmit={handleSendMessage} className="relative flex items-center bg-[#111111] border border-white/10 rounded-[2.5rem] p-2 focus-within:border-white/30 transition-all shadow-2xl">
              <button type="button" onClick={() => fileInputRef.current.click()} className="p-3 text-gray-500 hover:text-white transition-colors">
                <PaperClipIcon className="h-6 w-6" />
              </button>
              <input 
                type="file" 
                className="hidden" 
                ref={fileInputRef} 
                multiple // 다중 선택 허용
                onChange={(e) => {
                  const files = Array.from(e.target.files);
                  setSelectedFiles(prev => [...prev, ...files]); 
                }} 
              />
              
              <input 
                type="text" 
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Please upload the file to analyze in Ctrl and enter your question." 
                className="flex-1 bg-transparent py-3 px-4 focus:outline-none font-medium text-white text-xl"
              />
              
              <button 
                type="submit" 
                disabled={!inputText.trim() && selectedFiles.length === 0}
                className={`p-4 rounded-[2rem] transition-all shadow-lg ${
                  inputText || selectedFiles.length > 0 ? 'bg-white text-black hover:scale-105' : 'bg-[#222222] text-gray-600 cursor-not-allowed'
                }`}
              >
                <PaperAirplaneIcon className="h-7 w-7" />
              </button>
            </form>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;