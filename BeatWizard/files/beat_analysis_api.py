#!/usr/bin/env python3

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import json
from beat_analyzer import AudioFeatureExtractor
import uvicorn

app = FastAPI(title="Beat Analysis API")

# Enable CORS for Flutter web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-beat")
async def analyze_beat(audio_file: UploadFile = File(...)):
    """
    Analyze an audio file and return detailed audio features
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.filename.split('.')[-1]}") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Analyze the audio file
        extractor = AudioFeatureExtractor()
        features = extractor.extract_features(temp_file_path)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        # Return analysis as JSON
        return {
            "success": True,
            "analysis": features,
            "filename": audio_file.filename
        }
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Beat Analysis API is running"}

if __name__ == "__main__":
    print("🎵 Starting Beat Analysis API on http://localhost:8000")
    print("📖 API docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000) 