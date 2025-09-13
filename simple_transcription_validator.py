import os
import sys
import argparse
import asyncio
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from agentic_asr.tools.tool_validators import ToolValidator
from agentic_asr.tools.enhanced import (
    correct_transcription_tool,
    translate_transcription_file_tool
)


class SimpleTranscriptionProcessor:
    """Simplified transcription processor with async validation."""
    
    def __init__(self):
        self.validator = ToolValidator()
    
    def find_files(self, path: str) -> List[Path]:
        """Find transcription files."""
        path_obj = Path(path)
        
        if path_obj.is_file():
            return [path_obj] if path_obj.suffix.lower() in ['.txt', '.json', '.md'] else []
        
        if path_obj.is_dir():
            files = []
            for ext in ['*.txt', '*.json', '*.md']:
                files.extend(path_obj.glob(ext))
            return sorted(files)
        
        return []
    
    def _calculate_fallback_score(self, original_text: str, processed_text: str, operation: str) -> float:
        """Calculate a simple fallback score based on text metrics."""
        if not original_text or not processed_text:
            return 5.0
        
        # Simple scoring based on length and basic metrics
        length_ratio = len(processed_text) / len(original_text)
        
        # Reasonable length change suggests good processing
        if 0.8 <= length_ratio <= 1.3:
            base_score = 8.5
        elif 0.6 <= length_ratio <= 1.5:
            base_score = 7.5
        else:
            base_score = 6.5
        
        # Bonus for punctuation improvements (simple heuristic)
        original_punct = sum(1 for c in original_text if c in '.,!?;:')
        processed_punct = sum(1 for c in processed_text if c in '.,!?;:')
        
        if processed_punct > original_punct:
            base_score += 0.5  # Likely added proper punctuation
        
        # Cap at 10.0
        return min(base_score, 10.0)
    
    async def correct_file(self, file_path: Path) -> bool:
        """Correct a transcription file."""
        print(f"📝 Correcting: {file_path.name}")
        
        # Read content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print("  ❌ File is empty")
            return False
        
        # Apply correction
        result = await correct_transcription_tool(
            original_text=content,
            correction_level="medium"
        )
        
        if not result.get("success"):
            print(f"  ❌ Correction failed: {result.get('error', 'Unknown error')}")
            return False
        
        corrected_text = result["corrected_text"]
        
        # Perform validation and scoring
        try:
            validation_result = await self.validator.validate_correction(
                original_text=content,
                corrected_text=corrected_text,
                validation_level="basic"
            )
            score = validation_result["overall_score"]
            
            # Show some validation details
            if "scores" in validation_result:
                scores = validation_result["scores"]
                accuracy = scores.get("accuracy", 0)
                fluency = scores.get("fluency", 0)
                print(f"  📊 Accuracy: {accuracy:.1f}, Fluency: {fluency:.1f}")
                
        except Exception as e:
            print(f"  ⚠️  Validation error, using fallback: {str(e)[:50]}")
            score = self._calculate_fallback_score(content, corrected_text, "correction")
        
        print(f"  ✅ Corrected (Score: {score:.1f}/10)")
        
        # Save corrected version
        output_path = file_path.with_suffix('.corrected.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(corrected_text)
        print(f"  💾 Saved to: {output_path.name}")
        
        return True
    
    async def translate_file(self, file_path: Path, target_lang: str) -> bool:
        """Translate a transcription file."""
        print(f"🌐 Translating: {file_path.name} → {target_lang}")
        
        # Read content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print("  ❌ File is empty")
            return False
        
        # Apply translation
        result = await translate_transcription_file_tool(
            filename=file_path.name,
            target_language=target_lang
        )
        
        if not result.get("success"):
            print(f"  ❌ Translation failed: {result.get('error', 'Unknown error')}")
            return False
        
        translated_text = result["translated_text"]
        
        # Perform validation and scoring
        try:
            validation_result = await self.validator.validate_translation(
                original_text=content,
                translated_text=translated_text,
                source_language="auto-detect",
                target_language=target_lang,
                validation_level="basic"
            )
            score = validation_result["overall_score"]
            
            # Show some validation details
            if "scores" in validation_result:
                scores = validation_result["scores"]
                accuracy = scores.get("accuracy", 0)
                fluency = scores.get("fluency", 0)
                print(f"  📊 Accuracy: {accuracy:.1f}, Fluency: {fluency:.1f}")
                
        except Exception as e:
            print(f"  ⚠️  Validation error, using fallback: {str(e)[:50]}")
            score = self._calculate_fallback_score(content, translated_text, "translation")
        
        print(f"  ✅ Translated (Score: {score:.1f}/10)")
        
        # Save translated version
        output_path = file_path.with_suffix(f'.{target_lang}.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)
        print(f"  💾 Saved to: {output_path.name}")
        
        return True
    
    async def process_files(self, files: List[Path], mode: str, target_lang: str = "english"):
        """Process files based on mode."""
        print(f"🚀 Processing {len(files)} files in '{mode}' mode\n")
        
        successful = 0
        
        for i, file_path in enumerate(files, 1):
            print(f"[{i}/{len(files)}]", end=" ")
            
            if mode == "correct":
                success = await self.correct_file(file_path)
            elif mode == "translate":
                success = await self.translate_file(file_path, target_lang)
            else:
                print(f"❌ Unknown mode: {mode}")
                continue
            
            if success:
                successful += 1
            print()
        
        print(f"📊 Results: {successful}/{len(files)} files processed successfully")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Simplified transcription processing and validation"
    )
    
    # Input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Single file to process")
    group.add_argument("--path", type=str, help="Directory with transcription files")
    
    # Mode
    parser.add_argument("--mode", choices=["correct", "translate"], default="correct",
                       help="Processing mode (default: correct)")
    parser.add_argument("--target-lang", type=str, default="english",
                       help="Target language for translation (default: english)")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = SimpleTranscriptionProcessor()
    
    # Find files
    input_path = args.file or args.path
    files = processor.find_files(input_path)
    
    if not files:
        print(f"❌ No transcription files found in: {input_path}")
        sys.exit(1)
    
    print(f"🔍 Found {len(files)} file(s):")
    for file in files:
        print(f"  📄 {file.name}")
    print()
    
    # Process files
    await processor.process_files(files, args.mode, args.target_lang)


if __name__ == "__main__":
    asyncio.run(main())
