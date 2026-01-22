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
  confidence?: string; // Made optional to prevent crashes if model omits it
  reason?: string;
}

interface ProcessingStatus {
  step: 'idle' | 'extracting_pdf1' | 'extracting_pdf2' | 'extracting_video' | 'analyzing' | 'done' | 'error';
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
      const allPdfImages: PdfPageImage[] = [];

      // 1. Process PDF 1
      setStatus({ step: 'extracting_pdf1', message: '正在處理 PDF 1 (第一份簡報)...', progress: 5 });
      const images1 = await extractPdfImages(pdfFile1, 1);
      allPdfImages.push(...images1);
      
      // 2. Process PDF 2
      setStatus({ step: 'extracting_pdf2', message: '正在處理 PDF 2 (第二份簡報)...', progress: 15 });
      const images2 = await extractPdfImages(pdfFile2, 2);
      allPdfImages.push(...images2);

      setPdfImages(allPdfImages);
      
      // 3. Process Video
      setStatus({ step: 'extracting_video', message: '正在進行最佳化影片取樣 (700 幀)...', progress: 25 });
      
      // We no longer pass a fixed interval. The function calculates it based on video length.
      const extractedVideoFrames = await extractVideoFrames(videoFile, (progress) => {
         // Map 0-100 to 25-65 total progress
         setStatus(prev => ({ ...prev, progress: 25 + (progress * 0.4) })); 
      });
      setVideoFrames(extractedVideoFrames);

      // 4. Analyze with Gemini
      setStatus({ step: 'analyzing', message: '正在逐幀分析 (Gemini 2.5 Pro)...', progress: 70 });
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
      cMapUrl: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/cmaps/',
      cMapPacked: true,
    }).promise;
    
    const images: PdfPageImage[] = [];
    const totalPages = pdf.numPages;

    for (let i = 1; i <= totalPages; i++) {
      const page = await pdf.getPage(i);
      const viewport = page.getViewport({ scale: 1.0 });
      
      // Scale down: max width 512px (Reduced from 768px to save tokens)
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
          dataUrl: canvas.toDataURL('image/jpeg', 0.7) // Slightly reduced quality for token efficiency
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

        // --- Sampling Strategy ---
        // Set to 700 frames as requested.
        const TARGET_FRAME_COUNT = 700; 
        //const MIN_INTERVAL = 2; // seconds
        
        let interval =Math.max(2, Math.floor( duration / TARGET_FRAME_COUNT)); 
        //if (interval < MIN_INTERVAL) interval = MIN_INTERVAL;

        let currentTime = 0;
        
        // Optimization: Reduced max width to 256
        const scale = Math.min(1, 256 / Math.max(video.videoWidth, video.videoHeight));
        canvas.width = video.videoWidth * scale;
        canvas.height = video.videoHeight * scale;

        const processFrame = async () => {
          if (currentTime > duration) {
            // Ensure we capture near the end if we missed it by a large margin
            const lastFrameTime = frames.length > 0 ? frames[frames.length - 1].timestamp : -1;
            if (duration - lastFrameTime > interval * 1.5) {
                 video.currentTime = duration - 0.1; 
            }
             
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
               // Optimization: Low quality JPEG (0.3) for high volume data transmission
               dataUrl: canvas.toDataURL('image/jpeg', 0.3)
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
    
    // Split images by PDF ID for clarity in prompt
    const pdf1Images = pdfImgs.filter(img => img.pdfId === 1);
    const pdf2Images = pdfImgs.filter(img => img.pdfId === 2);

    parts.push({ text: "Here are the reference slides from the FIRST PDF Presentation (PDF 1). This content appears first in the video:" });
    
    pdf1Images.forEach(img => {
      parts.push({ text: `PDF 1 - Page ${img.pageNumber}` });
      parts.push({
        inlineData: {
          mimeType: "image/jpeg",
          data: img.dataUrl.split(',')[1]
        }
      });
    });

    parts.push({ text: "\n\nHere are the reference slides from the SECOND PDF Presentation (PDF 2). This content appears AFTER PDF 1 in the video:" });

    pdf2Images.forEach(img => {
      parts.push({ text: `PDF 2 - Page ${img.pageNumber}` });
      parts.push({
        inlineData: {
          mimeType: "image/jpeg",
          data: img.dataUrl.split(',')[1]
        }
      });
    });

    parts.push({ text: `\n\nHere are the frames extracted from the video (Sampled uniformly across duration):` });

    videoFrms.forEach(frm => {
      parts.push({ text: `Video Timestamp: ${frm.timeString}` });
      parts.push({
        inlineData: {
          mimeType: "image/jpeg",
          data: frm.dataUrl.split(',')[1]
        }
      });
    });

    parts.push({ text: `
      Analyze the video frames and determine which PDF page is currently visible in each frame.
      
      Important Context:
      - The video is a continuous recording of two presentations.
      - First, the speaker presents slides from PDF 1.
      - Then, the speaker switches to presenting slides from PDF 2.
      - A transition from PDF 1 to PDF 2 should happen exactly once.
      
      Output a list of "Transition Events". A transition event occurs when the slide changes.
      Include the very first slide at the beginning (Timestamp 00:00).
      
      For each transition, provide:
      1. The timestamp (MM:SS) where this slide FIRST appears.
      2. The ID of the PDF (1 or 2).
      3. The PDF Page Number.
      4. A confidence level (High/Medium/Low).
      
      Ensure you analyze the ENTIRE duration of the video frames provided.
      Strictly follow the JSON schema.
    ` });

    // Use gemini-2.5-flash as requested
    const modelId = "gemini-2.5-flash"; 

    const response = await ai.models.generateContent({
      model: modelId,
      contents: { parts },
      config: {
        responseMimeType: "application/json",
        maxOutputTokens: 8192, 
        // Simplified schema: Removed 'seconds' to avoid calculation errors by the LLM
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
                  confidence: { type: Type.STRING },
                }
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
      // Robust JSON Extraction & Repair
      let cleanText = responseText;

      // 1. Remove Markdown code blocks
      cleanText = cleanText.replace(/```json/g, '').replace(/```/g, '');

      // 2. Remove JS-style comments which invalidates JSON
      cleanText = cleanText.replace(/\/\/.*$/gm, ''); 
      cleanText = cleanText.replace(/\/\*[\s\S]*?\*\//g, '');

      // 3. Locate the outer JSON object braces using Regex for robust extraction
      const jsonMatch = cleanText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        cleanText = jsonMatch[0];
      }
      
      // 4. Fix missing commas between array objects (Common LLM error: } { -> }, {)
      cleanText = cleanText.replace(/}\s*{/g, '}, {');
      
      // 5. Balance Brackets (Handle truncated responses)
      const openBrackets = (cleanText.match(/\[/g) || []).length;
      const closeBrackets = (cleanText.match(/\]/g) || []).length;
      const openBraces = (cleanText.match(/{/g) || []).length;
      const closeBraces = (cleanText.match(/}/g) || []).length;

      if (closeBrackets < openBrackets) {
        cleanText += ']'.repeat(openBrackets - closeBrackets);
      }
      if (closeBraces < openBraces) {
        cleanText += '}'.repeat(openBraces - closeBraces);
      }

      // 6. Parse JSON
      const json = JSON.parse(cleanText);
      const rawTransitions = json.transitions || [];

      // --- Midpoint Correction Algorithm ---
      // Since we sample frames at an interval, the "detected" time is always late (the frame AFTER the change).
      // We correct this by shifting the timestamp back by half the sampling interval.
      // correctedTime = detectedTime - (avgInterval / 2)
      
      let avgInterval = 0;
      if (videoFrms.length > 1) {
         // Calculate actual interval from sampled frames
         avgInterval = videoFrms[1].timestamp - videoFrms[0].timestamp;
      } else if (videoFrms.length === 1) {
         avgInterval = 0; 
      }

      const correctionFactor = avgInterval / 2;

      const correctedTransitions = rawTransitions.map((t: any) => {
          // Manually parse timestamp to seconds since we removed 'seconds' from schema
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
      throw new Error("無法解析分析結果。模型回應格式錯誤。請重試。");
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="flex items-center space-x-3 border-b border-slate-800 pb-6">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Play className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
              簡報同步 AI (雙份簡報)
            </h1>
            <p className="text-slate-400 text-sm">
              將影片時間軸與兩份連續的 PDF 簡報（第一部分 &rarr; 第二部分）同步。
            </p>
          </div>
        </div>

        {/* Input Section */}
        {status.step === 'idle' || status.step === 'error' ? (
          <div className="grid md:grid-cols-2 gap-6 h-full">
            
            {/* Left Column: Video Input (Full Height) */}
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

            {/* Right Column: 2 PDF Inputs (Stacked) */}
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
                  <p className="text-slate-400 text-xs">
                    從影片開頭開始
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
                  <p className="text-slate-400 text-xs">
                    接續在 PDF 1 之後
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
                        <th className="p-4 w-32">時間</th>
                        <th className="p-4 w-24 text-center">來源</th>
                        <th className="p-4 w-24 text-center">頁碼</th>
                        <th className="p-4">投影片預覽</th>
                        <th className="p-4 w-32">信心指數</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {results.map((match, idx) => {
                        // Find the PDF image for preview
                        const pdfImg = pdfImages.find(p => p.pageNumber === match.pageNumber && p.pdfId === match.pdfId);
                        
                        return (
                          <tr key={idx} className="hover:bg-slate-800/30 transition-colors group">
                            <td className="p-4 font-mono text-lg font-medium text-blue-400 flex items-center space-x-2">
                               <Clock className="w-4 h-4 text-slate-600" />
                               <span>{match.timestamp}</span>
                            </td>
                             <td className="p-4 text-center">
                               <div className={`inline-block px-2 py-1 rounded text-xs font-bold uppercase ${match.pdfId === 1 ? 'bg-red-500/20 text-red-400' : 'bg-orange-500/20 text-orange-400'}`}>
                                 PDF {match.pdfId}
                               </div>
                            </td>
                            <td className="p-4 text-center">
                               <div className="inline-block bg-slate-800 rounded px-2 py-1 font-bold">
                                 #{match.pageNumber}
                               </div>
                            </td>
                            <td className="p-4">
                               {pdfImg ? (
                                 <div className="flex items-start space-x-4">
                                   <img 
                                     src={pdfImg.dataUrl} 
                                     alt={`PDF ${match.pdfId} - Page ${match.pageNumber}`} 
                                     className="h-24 rounded border border-slate-700 shadow-sm" 
                                   />
                                   {match.reason && (
                                     <p className="text-sm text-slate-500 max-w-md hidden md:block italic">
                                       "{match.reason}"
                                     </p>
                                   )}
                                 </div>
                               ) : (
                                 <span className="text-slate-600 italic">無預覽</span>
                               )}
                            </td>
                            <td className="p-4">
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