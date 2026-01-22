import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type } from "@google/genai";
import { Upload, FileVideo, FileText, Play, Loader2, CheckCircle, AlertCircle, Clock, Image as ImageIcon, ArrowDown } from 'lucide-react';

// --- Types ---

interface SlideMatch {
  timestamp: string;
  seconds: number;
  pdfId: number; // 1 or 2
  pageNumber: number;
  slideTitle: string; // New field
  reasoning: string;  // New field
  confidence: string; // "High" | "Medium" | "Low"
}

interface ProcessingStatus {
  step: 'idle' | 'extracting' | 'analyzing' | 'done' | 'error';
  message: string;
  progress: number; // 0 to 100
}

interface PdfPageImage {
  pdfId: number;
  pageNumber: number;
  dataUrl: string; // base64
}

interface VideoFrameImage {
  timestamp: number;
  timeString: string;
  dataUrl: string; // base64
}

// --- Helper Functions ---

const formatTime = (seconds: number): string => {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
};

const parseTimeToSeconds = (timeStr: string): number => {
  if (!timeStr) return 0;
  // Handle MM:SS or HH:MM:SS
  const parts = timeStr.split(':').map(part => parseInt(part.trim(), 10));
  
  if (parts.length === 2) {
    return parts[0] * 60 + parts[1];
  } else if (parts.length === 3) {
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  return 0;
};

// --- Components ---

const App = () => {
  const [apiKey, setApiKey] = useState(process.env.API_KEY || '');
  
  // File States
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [pdfFile1, setPdfFile1] = useState<File | null>(null);
  const [pdfFile2, setPdfFile2] = useState<File | null>(null);
  
  // Data States
  const [pdfImages, setPdfImages] = useState<PdfPageImage[]>([]);
  const [videoFrames, setVideoFrames] = useState<VideoFrameImage[]>([]);
  const [results, setResults] = useState<SlideMatch[]>([]);
  
  // Status State
  const [status, setStatus] = useState<ProcessingStatus>({ step: 'idle', message: '', progress: 0 });

  const processFiles = async () => {
    if (!videoFile || !pdfFile1 || !pdfFile2 || !apiKey) return;

    try {
      setStatus({ step: 'extracting', message: '正在平行處理檔案 (影片與 PDF)...', progress: 5 });

      // 1. Parallel Processing: Extract all data simultaneously
      const [images1, images2, extractedVideoFrames] = await Promise.all([
        extractPdfImages(pdfFile1, 1),
        extractPdfImages(pdfFile2, 2),
        extractVideoFrames(videoFile, (p) => {
           // Update progress based on video extraction (usually the longest task)
           // Map video progress 0-100 to overall progress 10-60
           setStatus(prev => ({ ...prev, progress: 10 + (p * 0.5) }));
        })
      ]);

      const allPdfImages = [...images1, ...images2];
      setPdfImages(allPdfImages);
      setVideoFrames(extractedVideoFrames);

      // 2. Analyze with Gemini
      setStatus({ step: 'analyzing', message: '正在進行多模態 AI 分析 (Gemini 2.5 Flash)...', progress: 70 });
      const analysisResults = await analyzeWithGemini(allPdfImages, extractedVideoFrames);
      
      setResults(analysisResults);
      setStatus({ step: 'done', message: '分析完成！', progress: 100 });

    } catch (error: any) {
      console.error(error);
      setStatus({ step: 'error', message: error.message || '發生錯誤', progress: 0 });
    }
  };

  const extractPdfImages = async (file: File, pdfId: number): Promise<PdfPageImage[]> => {
    const arrayBuffer = await file.arrayBuffer();
    // @ts-ignore - pdfjsLib is loaded globally in index.html
    const pdf = await window.pdfjsLib.getDocument({ 
      data: arrayBuffer,
      // Use unpkg for reliable cMap serving to fix font issues
      cMapUrl: 'https://unpkg.com/pdfjs-dist@3.11.174/cmaps/',
      cMapPacked: true,
    }).promise;
    
    const images: PdfPageImage[] = [];
    const totalPages = pdf.numPages;

    for (let i = 1; i <= totalPages; i++) {
      const page = await pdf.getPage(i);
      const viewport = page.getViewport({ scale: 1.0 });
      
      // Scale down: max width 512px
      const scale = Math.min(1.0, 512 / viewport.width);
      const scaledViewport = page.getViewport({ scale });

      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      canvas.height = scaledViewport.height;
      canvas.width = scaledViewport.width;

      if (context) {
        await page.render({ canvasContext: context, viewport: scaledViewport }).promise;
        images.push({
          pdfId: pdfId,
          pageNumber: i,
          dataUrl: canvas.toDataURL('image/jpeg', 0.7)
        });
      }
    }
    return images;
  };

  const extractVideoFrames = async (file: File, onProgress: (p: number) => void): Promise<VideoFrameImage[]> => {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      video.src = URL.createObjectURL(file);
      video.muted = true;
      video.playsInline = true;

      const frames: VideoFrameImage[] = [];
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      video.onloadedmetadata = async () => {
        const duration = video.duration;
        
        if (!Number.isFinite(duration)) {
           reject(new Error("無法確定影片長度。"));
           return;
        }

        // --- Sampling Strategy (Strict Control) ---
        // Maintain 500 frames as requested for high precision.
        const TARGET_FRAME_COUNT = 500; 
        let interval = duration / TARGET_FRAME_COUNT;
        // Ensure strictly minimal interval to avoid duplicates if video is short
        if (interval < 0.1) interval = 0.1;

        let currentTime = 0;
        
        // Optimization: Limit to 256px width
        const scale = Math.min(1.0, 256 / video.videoWidth);
        canvas.width = video.videoWidth * scale;
        canvas.height = video.videoHeight * scale;

        const processFrame = async () => {
          if (currentTime > duration) {
            URL.revokeObjectURL(video.src);
            resolve(frames);
            return;
          }
          video.currentTime = currentTime;
        };

        video.onseeked = () => {
          if (ctx) {
             ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
             frames.push({
               timestamp: currentTime,
               timeString: formatTime(currentTime),
               // Low quality JPEG for efficient token usage with high frame count
               dataUrl: canvas.toDataURL('image/jpeg', 0.5) 
             });
          }
          
          onProgress((currentTime / duration) * 100);
          currentTime += interval;
          processFrame();
        };
        
        video.onerror = (e) => reject(new Error("影片播放錯誤"));
        
        // Start processing
        processFrame();
      };

      video.onerror = (e) => reject(new Error("無法載入影片"));
    });
  };

  const analyzeWithGemini = async (pdfImgs: PdfPageImage[], videoFrms: VideoFrameImage[]): Promise<SlideMatch[]> => {
    const ai = new GoogleGenAI({ apiKey });
    
    const parts: any[] = [];
    
    // 1. PDF 1 Section
    parts.push({ text: "【參考資料 A：第一份簡報 (PDF 1)】\n這是演講上半場使用的投影片，依序出現：" });
    const pdf1Images = pdfImgs.filter(img => img.pdfId === 1);
    pdf1Images.forEach(img => {
      parts.push({ text: `(PDF1 Page ${img.pageNumber})` });
      parts.push({
        inlineData: {
          mimeType: "image/jpeg",
          data: img.dataUrl.split(',')[1]
        }
      });
    });

    // 2. PDF 2 Section
    parts.push({ text: "\n\n【參考資料 B：第二份簡報 (PDF 2)】\n這是演講下半場使用的投影片，會接續在 PDF 1 之後出現：" });
    const pdf2Images = pdfImgs.filter(img => img.pdfId === 2);
    pdf2Images.forEach(img => {
      parts.push({ text: `(PDF2 Page ${img.pageNumber})` });
      parts.push({
        inlineData: {
          mimeType: "image/jpeg",
          data: img.dataUrl.split(',')[1]
        }
      });
    });

    // 3. Video Frames Section
    parts.push({ text: "\n\n【待分析目標：影片影格序列】\n以下是從演講影片中按時間順序取樣的畫面 (帶有時間戳記)：" });
    videoFrms.forEach(frm => {
      parts.push({ text: `\n[VIDEO_TIMESTAMP: ${frm.timeString}]` });
      parts.push({
        inlineData: {
          mimeType: "image/jpeg",
          data: frm.dataUrl.split(',')[1]
        }
      });
    });

    // 4. System Prompt (Optimized)
    const systemPrompt = `
"""
你是一位專業的演講影片分析專家，擅長將「現場演講影片」與「原始 PDF 投影片」進行視覺同步。

**任務目標：**
分析提供的【影片影格序列】，找出每一張投影片（來自 PDF 1 或 PDF 2）在影片中「首次清晰出現」的時間點。

**核心分析邏輯與規則 (請嚴格遵守)：**

1.  **忽略非投影片畫面 (抗干擾)**：
    * 影片開頭通常包含主持人介紹、講者特寫或等待畫面。請務必等到**投影片內容清晰充滿畫面**，且與 PDF 某頁高度相符時，才標記第一個事件。
    * **切勿強行從 00:00 開始**，除非 00:00 確實就是投影片畫面。
    * 若畫面中只有講者、觀眾或過場動畫，請**忽略**該影格，不要強行匹配。

2.  **雙份簡報切換邏輯 (PDF 1 -> PDF 2)**：
    * 影片內容是連續的。順序必然是：先展示 PDF 1 的頁面 -> (可能有一段講者串場/休息) -> 接著展示 PDF 2 的頁面。
    * PDF 1 與 PDF 2 之間**只會發生一次**切換。
    * 在切換期間（例如換檔空檔），若畫面無投影片，請勿產生匹配事件。

3.  **視覺匹配優先**：
    * 請根據畫面中的文字標題、圖表形狀、圖片排版進行比對。
    * **標題識別**：請優先讀取投影片上方的大字體標題作為 \`slideTitle\`。若無標題，請總結畫面核心內容。

4.  **輸出格式 (JSON)**：
    請輸出一個 JSON 物件，包含一個 \`transitions\` 陣列。每個元素代表一次「投影片更換事件」。
    格式要求：
    * \`timestamp\`: 字串 (MM:SS)，該投影片**首次**出現的精確時間。
    * \`pdfId\`: 整數 (1 或 2)，代表屬於哪一份 PDF。
    * \`pageNumber\`: 整數，對應的 PDF 頁碼。
    * \`slideTitle\`: 字串，投影片標題。
    * \`reasoning\`: 字串 (繁體中文)，簡述判斷理由 (例如：「畫面標題與 PDF 1 第 3 頁一致」、「圖表吻合」)。請盡量保持 reasoning 簡短，以避免輸出長度超出 Token 限制。
    * \`confidence\`: 字串 ("High", "Medium", "Low")。
"""
`;
    parts.push({ text: systemPrompt });

    // Use gemini-2.5-flash as requested
    const modelId = "gemini-2.5-flash"; 

    const response = await ai.models.generateContent({
      model: modelId,
      contents: { parts },
      config: {
        responseMimeType: "application/json",
        maxOutputTokens: 8192, 
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            transitions: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  timestamp: { type: Type.STRING, description: "Time format MM:SS" },
                  pdfId: { type: Type.INTEGER, description: "1 for PDF 1, 2 for PDF 2" },
                  pageNumber: { type: Type.INTEGER },
                  slideTitle: { type: Type.STRING, description: "Title of the slide identified" },
                  reasoning: { type: Type.STRING, description: "Why this match was made. Keep it short." },
                  confidence: { type: Type.STRING, enum: ["High", "Medium", "Low"] },
                },
                required: ["timestamp", "pdfId", "pageNumber", "slideTitle", "reasoning", "confidence"]
              }
            }
          }
        }
      }
    });

    const responseText = response.text;
    if (!responseText) {
      throw new Error("模型回傳了空的回應。這可能是由於內容安全過濾器所致。");
    }

    try {
      // --- Robust "Discard Incomplete Tail" JSON Parsing ---
      let cleanText = responseText;

      // 1. Remove Markdown code blocks
      cleanText = cleanText.replace(/```json/g, '').replace(/```/g, '').trim();

      // 2. Remove JS-style comments (just in case)
      cleanText = cleanText.replace(/\/\/.*$/gm, ''); 
      cleanText = cleanText.replace(/\/\*[\s\S]*?\*\//g, '');

      // 3. Locate the outer JSON object braces
      const jsonStart = cleanText.indexOf('{');
      if (jsonStart !== -1) {
        cleanText = cleanText.substring(jsonStart);
      }

      // 4. Truncation Repair Strategy
      // Check if it ends with a proper closure. If not, discard the tail.
      // We expect the array of transitions to be the main content.
      // If it doesn't end with '}', it's likely truncated.
      if (!cleanText.endsWith('}')) {
          console.warn("Response appeared truncated. Applying discard logic.");
          
          // Find the last complete object in the array. 
          // The structure is { "transitions": [ {...}, {...}, ... ] }
          // An item always ends with '},' if there is another item following, 
          // or '}' if it's the last item (but here it's truncated, so that '}' might be missing or inside a string).
          
          // We look for '},' which signifies the end of a completed object and the start of the next (incomplete) one.
          const lastValidObjectEnd = cleanText.lastIndexOf('},');
          
          if (lastValidObjectEnd !== -1) {
              // Keep everything up to the closing brace '}' of the last valid object.
              // '},' is 2 chars. We want to keep the '}'. So we take up to index + 1.
              cleanText = cleanText.substring(0, lastValidObjectEnd + 1);
              
              // Close the array and the root object manually
              cleanText += ']}';
          } else {
             // Fallback: If we can't find '},', it might mean only 1 item exists and it's truncated,
             // or the array is empty. This is a severe truncation case.
             // We can try to see if it starts with { "transitions": [ and just close it empty to avoid crash.
             if (cleanText.includes('"transitions": [')) {
                 // Try to close it as empty or valid-ish if possible, but safer to error or return empty array logic.
                 // Let's assume if we can't find a single completed object separator, we might as well treat it as empty result
                 // to prevent parsing errors.
                 console.warn("Could not find any complete transition objects. Returning empty list.");
                 cleanText = '{ "transitions": [] }';
             }
          }
      }

      const json = JSON.parse(cleanText);
      const rawTransitions = json.transitions || [];

      // --- Midpoint Correction Algorithm ---
      // correctedTime = detectedTime - (avgInterval / 2)
      let avgInterval = 0;
      if (videoFrms.length > 1) {
         avgInterval = videoFrms[1].timestamp - videoFrms[0].timestamp;
      }

      const correctionFactor = avgInterval / 2;

      const correctedTransitions = rawTransitions.map((t: any) => {
          const detectedSeconds = parseTimeToSeconds(t.timestamp);
          let correctedSeconds = detectedSeconds - correctionFactor;
          if (correctedSeconds < 0) correctedSeconds = 0;
          
          return {
            ...t,
            seconds: correctedSeconds,
            timestamp: formatTime(correctedSeconds)
          };
      });

      return correctedTransitions;

    } catch (e) {
      console.error("JSON Parsing Error:", e);
      console.log("Raw Model Output:", responseText);
      throw new Error("無法解析分析結果。模型回應格式錯誤 (JSON Error)。");
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="flex items-center space-x-3 border-b border-slate-800 pb-6">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Play className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
              簡報同步 AI (雙份簡報 - 高精度版)
            </h1>
            <p className="text-slate-400 text-sm">
              平行處理 | 智慧識別 | 時間軸修正
            </p>
          </div>
        </div>

        {/* Input Section */}
        {status.step === 'idle' || status.step === 'error' ? (
          <div className="grid md:grid-cols-2 gap-6 h-full">
            
            {/* Left Column: Video Input */}
            <div className={`
              h-full min-h-[400px] border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center space-y-4 transition-colors
              ${videoFile ? 'border-blue-500 bg-blue-500/10' : 'border-slate-700 hover:border-slate-500 hover:bg-slate-900'}
            `}>
              <div className="bg-slate-800 p-4 rounded-full">
                <FileVideo className="w-10 h-10 text-blue-400" />
              </div>
              <div className="text-center">
                <p className="font-medium text-xl">
                  {videoFile ? videoFile.name : "上傳完整影片"}
                </p>
                <p className="text-slate-400 text-sm mt-1">
                   {videoFile ? `${(videoFile.size / 1024 / 1024).toFixed(2)} MB` : "MP4, MOV, WebM"}
                </p>
              </div>
              <label className="cursor-pointer">
                <input 
                  type="file" 
                  accept="video/*" 
                  className="hidden" 
                  onChange={(e) => setVideoFile(e.target.files?.[0] || null)} 
                />
                <span className="px-6 py-3 bg-slate-800 hover:bg-slate-700 rounded-md font-medium transition-colors">
                  {videoFile ? "更換影片" : "選擇影片檔案"}
                </span>
              </label>
            </div>

            {/* Right Column: 2 PDF Inputs */}
            <div className="flex flex-col space-y-6">
              
              {/* PDF 1 Input */}
              <div className={`
                flex-1 border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center space-y-3 transition-colors relative
                ${pdfFile1 ? 'border-red-500 bg-red-500/10' : 'border-slate-700 hover:border-slate-500 hover:bg-slate-900'}
              `}>
                <div className="absolute top-4 left-4 bg-slate-800 px-2 py-1 rounded text-xs font-bold text-slate-300 uppercase">
                  第一部分
                </div>
                <div className="bg-slate-800 p-3 rounded-full">
                  <FileText className="w-6 h-6 text-red-400" />
                </div>
                <div className="text-center">
                  <p className="font-medium text-lg">
                    {pdfFile1 ? pdfFile1.name : "上傳 PDF 1"}
                  </p>
                </div>
                <label className="cursor-pointer">
                  <input 
                    type="file" 
                    accept="application/pdf" 
                    className="hidden" 
                    onChange={(e) => setPdfFile1(e.target.files?.[0] || null)} 
                  />
                  <span className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-md text-sm font-medium transition-colors">
                    {pdfFile1 ? "更換 PDF 1" : "選擇 PDF 1"}
                  </span>
                </label>
              </div>

              {/* Arrow Indicator */}
              <div className="flex justify-center -my-3 z-10">
                 <ArrowDown className="w-6 h-6 text-slate-600" />
              </div>

              {/* PDF 2 Input */}
              <div className={`
                flex-1 border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center space-y-3 transition-colors relative
                ${pdfFile2 ? 'border-orange-500 bg-orange-500/10' : 'border-slate-700 hover:border-slate-500 hover:bg-slate-900'}
              `}>
                 <div className="absolute top-4 left-4 bg-slate-800 px-2 py-1 rounded text-xs font-bold text-slate-300 uppercase">
                  第二部分
                </div>
                <div className="bg-slate-800 p-3 rounded-full">
                  <FileText className="w-6 h-6 text-orange-400" />
                </div>
                <div className="text-center">
                  <p className="font-medium text-lg">
                    {pdfFile2 ? pdfFile2.name : "上傳 PDF 2"}
                  </p>
                </div>
                <label className="cursor-pointer">
                  <input 
                    type="file" 
                    accept="application/pdf" 
                    className="hidden" 
                    onChange={(e) => setPdfFile2(e.target.files?.[0] || null)} 
                  />
                  <span className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-md text-sm font-medium transition-colors">
                    {pdfFile2 ? "更換 PDF 2" : "選擇 PDF 2"}
                  </span>
                </label>
              </div>

            </div>

            {/* Analyze Button */}
            <div className="md:col-span-2 flex justify-center pt-4">
              <button
                disabled={!videoFile || !pdfFile1 || !pdfFile2}
                onClick={processFiles}
                className={`
                  flex items-center space-x-2 px-8 py-4 rounded-lg text-lg font-bold transition-all w-full md:w-auto justify-center
                  ${!videoFile || !pdfFile1 || !pdfFile2
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                    : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/50 scale-100 hover:scale-105 active:scale-95'}
                `}
              >
                <Play className="w-6 h-6 fill-current" />
                <span>開始雙份簡報同步分析</span>
              </button>
            </div>
            
            {status.step === 'error' && (
               <div className="md:col-span-2 bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-lg flex items-center space-x-3">
                 <AlertCircle className="w-5 h-5" />
                 <span>{status.message}</span>
               </div>
            )}

          </div>
        ) : (
          // Processing & Results View
          <div className="space-y-8 animate-in fade-in duration-500">
            
            {/* Progress Bar */}
            {status.step !== 'done' && (
              <div className="bg-slate-900 rounded-xl p-8 border border-slate-800 text-center space-y-4">
                <div className="relative w-16 h-16 mx-auto">
                   <Loader2 className="w-16 h-16 text-blue-500 animate-spin" />
                </div>
                <h2 className="text-xl font-medium">{status.message}</h2>
                <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                  <div 
                    className="bg-blue-500 h-full transition-all duration-300 ease-out" 
                    style={{ width: `${status.progress}%` }}
                  ></div>
                </div>
              </div>
            )}

            {/* Results Table */}
            {results.length > 0 && (
              <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
                <div className="p-6 border-b border-slate-800 flex justify-between items-center">
                  <h2 className="text-xl font-bold flex items-center space-x-2">
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    <span>分析結果</span>
                  </h2>
                  <button 
                    onClick={() => {
                        setResults([]);
                        setStatus({ step: 'idle', message: '', progress: 0 });
                    }}
                    className="text-slate-400 hover:text-white text-sm"
                  >
                    重新開始
                  </button>
                </div>
                
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="bg-slate-800/50 text-slate-400 text-sm uppercase tracking-wider">
                        <th className="p-4 w-24">時間</th>
                        <th className="p-4 w-20 text-center">來源</th>
                        <th className="p-4 w-16 text-center">頁碼</th>
                        <th className="p-4 w-64">投影片標題</th>
                        <th className="p-4 w-48">預覽</th>
                        <th className="p-4">AI 判斷理由</th>
                        <th className="p-4 w-24">信心度</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {results.map((match, idx) => {
                        // Find the PDF image for preview
                        const pdfImg = pdfImages.find(p => p.pageNumber === match.pageNumber && p.pdfId === match.pdfId);
                        
                        return (
                          <tr key={idx} className="hover:bg-slate-800/30 transition-colors group">
                            <td className="p-4 font-mono text-lg font-medium text-blue-400 align-top">
                               <div className="flex items-center space-x-2">
                                  <Clock className="w-4 h-4 text-slate-600" />
                                  <span>{match.timestamp}</span>
                               </div>
                            </td>
                             <td className="p-4 text-center align-top">
                               <div className={`inline-block px-2 py-1 rounded text-xs font-bold uppercase ${match.pdfId === 1 ? 'bg-red-500/20 text-red-400' : 'bg-orange-500/20 text-orange-400'}`}>
                                 PDF {match.pdfId}
                               </div>
                            </td>
                            <td className="p-4 text-center align-top">
                               <div className="inline-block bg-slate-800 rounded px-2 py-1 font-bold">
                                 #{match.pageNumber}
                               </div>
                            </td>
                            <td className="p-4 font-bold text-slate-200 align-top">
                                {match.slideTitle || "無標題"}
                            </td>
                            <td className="p-4 align-top">
                               {pdfImg ? (
                                   <img 
                                     src={pdfImg.dataUrl} 
                                     alt={`PDF ${match.pdfId} - Page ${match.pageNumber}`} 
                                     className="w-40 rounded border border-slate-700 shadow-sm" 
                                   />
                               ) : (
                                 <span className="text-slate-600 italic">無預覽</span>
                               )}
                            </td>
                            <td className="p-4 text-slate-400 text-sm align-top">
                                {match.reasoning || "無詳細說明"}
                            </td>
                            <td className="p-4 align-top">
                              <span className={`
                                px-2 py-1 rounded-full text-xs font-medium border
                                ${(match.confidence || 'low').toLowerCase() === 'high' 
                                  ? 'bg-green-500/10 text-green-400 border-green-500/20' 
                                  : 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'}
                              `}>
                                {match.confidence || '未知'}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);