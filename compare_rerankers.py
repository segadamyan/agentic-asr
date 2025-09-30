#!/usr/bin/env python3
"""
Comprehensive reranker comparison script.
Retrieves data from database, tests all rerankers, and shows comparison table.
"""

import logging
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from tabulate import tabulate
import sys
import os

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_asr.config import Config
from agentic_asr.rerankers import get_reranker

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during comparison
logger = logging.getLogger(__name__)


class RerankerComparator:
    """Compare different rerankers with real data from database."""
    
    def __init__(self):
        self.db_path = Config.DATABASE_PATH
        self.vector_store = None
        self.test_queries = []
        
    def load_data_from_vector_store(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Load data from existing vector store."""
        print("Loading data from existing vector store...")
        
        try:
            from agentic_asr.vector_store import get_vector_store
            
            # Initialize vector store
            vector_store = get_vector_store()
            vector_store.initialize()
            
            # Get statistics
            stats = vector_store.get_stats()
            print(f"Vector store contains {stats['total_documents']} documents with {stats['total_chunks']} chunks")
            
            if stats['total_chunks'] == 0:
                print("No data in vector store. Using AI-related sample data.")
                return self._create_ai_sample_data()
            
            # Use AI-related queries to get relevant chunks
            ai_queries = [
                "արհեստական բանականություն",  # artificial intelligence in Armenian
                "տեխնոլոգիա",  # technology
                "ծրագրավորում",  # programming
                "ալգորիթմ",  # algorithm
                "նեյրոնային ցանց",  # neural network
                "մեքենայական ուսուցում",  # machine learning
                "գիտություն",  # science
                "հետազոտություն"  # research
            ]
            
            results = []
            chunks_per_query = max(1, limit // len(ai_queries))
            
            for query in ai_queries:
                try:
                    search_results = vector_store.search(
                        query=query,
                        top_k=chunks_per_query,
                        similarity_threshold=0.1,
                        use_reranking=False  # We'll test reranking separately
                    )
                    
                    for result in search_results:
                        # Create a display name that includes chunk info
                        filename = result.get("filename", "unknown")
                        chunk_id = result.get("chunk_id", "unknown")
                        display_name = f"{filename}[{chunk_id}]"
                        
                        results.append({
                            "chunk_id": chunk_id,
                            "content": result["content"],
                            "filename": filename,
                            "display_name": display_name,
                            "language": "armenian",  # Based on your data
                            "similarity_score": result["similarity_score"]
                        })
                        
                        if len(results) >= limit:
                            break
                            
                except Exception as e:
                    print(f"Error searching for '{query}': {e}")
                    continue
                
                if len(results) >= limit:
                    break
            
            if not results:
                print("No relevant AI content found. Using sample data.")
                return self._create_ai_sample_data()
            
            print(f"Loaded {len(results)} AI-related chunks from vector store")
            return results[:limit]
            
        except Exception as e:
            print(f"Error accessing vector store: {e}")
            return self._create_ai_sample_data()
    
    def _create_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _create_ai_sample_data(self) -> List[Dict[str, Any]]:
        """Create AI-related sample data for testing."""
        print("Using AI-related sample data for comparison...")
        return [
            {
                "chunk_id": "1",
                "content": "Արհեստական բանականությունը և մեքենայական ուսուցումը ժամանակակից տեխնոլոգիաների հիմքն են: Նեյրոնային ցանցերը կարող են սովորել բարդ օրինաչափություններ տվյալներից:",
                "similarity_score": 0.95,
                "filename": "ai_basics_armenian.txt",
                "display_name": "ai_basics_armenian.txt[1]",
                "language": "armenian"
            },
            {
                "chunk_id": "2",
                "content": "Deep learning algorithms use neural networks with multiple layers to process and analyze large amounts of data. These systems can recognize patterns and make predictions with high accuracy.",
                "similarity_score": 0.92,
                "filename": "deep_learning_english.txt",
                "display_name": "deep_learning_english.txt[2]",
                "language": "english"
            },
            {
                "chunk_id": "3",
                "content": "Գիտական հետազոտությունները ցույց են տալիս, որ արհեստական ինտելեկտը կարող է օգտագործվել բժշկության, կրթության և գիտության ոլորտներում: Ալգորիթմները զարգանում են շատ արագ:",
                "similarity_score": 0.89,
                "filename": "ai_research_armenian.txt",
                "display_name": "ai_research_armenian.txt[3]",
                "language": "armenian"
            },
            {
                "chunk_id": "4",
                "content": "Natural language processing enables computers to understand and generate human language. This technology powers chatbots, translation services, and text analysis tools.",
                "similarity_score": 0.86,
                "filename": "nlp_technology.txt",
                "display_name": "nlp_technology.txt[4]",
                "language": "english"
            },
            {
                "chunk_id": "5",
                "content": "Ծրագրավորման նոր մեթոդները թույլ են տալիս ստեղծել ավելի արդյունավետ ալգորիթմներ: Python և R լեզուները լայնորեն օգտագործվում են տվյալների գիտության մեջ:",
                "similarity_score": 0.83,
                "filename": "programming_data_science.txt",
                "display_name": "programming_data_science.txt[5]",
                "language": "armenian"
            },
            {
                "chunk_id": "6",
                "content": "Computer vision technology allows machines to interpret and understand visual information from the world. Applications include facial recognition, autonomous vehicles, and medical imaging.",
                "similarity_score": 0.80,
                "filename": "computer_vision.txt",
                "display_name": "computer_vision.txt[6]",
                "language": "english"
            },
            {
                "chunk_id": "7",
                "content": "Տեխնոլոգիական ինովացիաները փոխում են մեր առօրյա կյանքը: Արհեստական բանականությունը օգնում է լուծել բարդ խնդիրներ տարբեր ոլորտներում:",
                "similarity_score": 0.77,
                "filename": "tech_innovation_armenian.txt",
                "display_name": "tech_innovation_armenian.txt[7]",
                "language": "armenian"
            },
            {
                "chunk_id": "8",
                "content": "Machine learning models require large datasets for training. Feature engineering and data preprocessing are crucial steps in developing accurate predictive models.",
                "similarity_score": 0.74,
                "filename": "ml_training_data.txt",
                "display_name": "ml_training_data.txt[8]",
                "language": "english"
            },
            {
                "chunk_id": "9",
                "content": "Գիտության և տեխնոլոգիաների զարգացումը պահանջում է մասնագիտական գիտելիքներ և անընդհատ ուսուցում: Քվանտային հաշվարկները նոր հնարավորություններ են բացում:",
                "similarity_score": 0.71,
                "filename": "quantum_computing_armenian.txt",
                "display_name": "quantum_computing_armenian.txt[9]",
                "language": "armenian"
            },
            {
                "chunk_id": "10",
                "content": "Artificial intelligence ethics and responsible AI development are becoming increasingly important as these technologies impact society. Bias detection and fairness in algorithms are key concerns.",
                "similarity_score": 0.68,
                "filename": "ai_ethics.txt",
                "display_name": "ai_ethics.txt[10]",
                "language": "english"
            }
        ]
    
    def get_test_queries(self) -> List[Dict[str, str]]:
        """Get AI-related test queries in different languages."""
        return [
            {
                "query": "արհեստական բանականություն և մեքենայական ուսուցում",
                "language": "armenian",
                "description": "Armenian: artificial intelligence and machine learning"
            },
            {
                "query": "neural networks deep learning algorithms",
                "language": "english", 
                "description": "English: neural networks and deep learning"
            },
            {
                "query": "ծրագրավորման լեզուներ և ալգորիթմներ",
                "language": "armenian",
                "description": "Armenian: programming languages and algorithms"
            },
            {
                "query": "computer vision natural language processing",
                "language": "english",
                "description": "English: computer vision and NLP"
            },
            {
                "query": "տեխնոլոգիական ինովացիա և գիտություն",
                "language": "armenian", 
                "description": "Armenian: technological innovation and science"
            },
            {
                "query": "machine learning model training data",
                "language": "english",
                "description": "English: ML model training"
            },
            {
                "query": "գիտական հետազոտություններ և տվյալներ",
                "language": "armenian",
                "description": "Armenian: scientific research and data"
            }
        ]
    
    def get_available_rerankers(self) -> List[Dict[str, Any]]:
        """Get list of available rerankers with their configs."""
        rerankers = [
            {"type": "none", "kwargs": {}, "name": "No Reranker"},
            {"type": "bm25", "kwargs": {}, "name": "BM25"},
            {"type": "keyword_boost", "kwargs": {"boost_factor": 1.5}, "name": "Keyword Boost"},
        ]
        
        rerankers.extend([
            {
                "type": "cross_encoder", 
                "kwargs": {"model_name": "cross-encoder/ms-marco-TinyBERT-L-2-v2"},
                "name": "Cross Encoder"
            },
            {
                "type": "multilingual_cross_encoder",
                "kwargs": {
                    "model_name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                    "armenian_boost": 1.2
                },
                "name": "Multilingual Cross Encoder"
            }
        ])
        
        # Check if LLM is available
        try:
            from agentic_asr.config import Config
            if Config.ANTHROPIC_API_KEY or Config.OPENAI_API_KEY:
                rerankers.append({
                    "type": "llm_agent",
                    "kwargs": {
                        "provider_name": "anthropic" if Config.ANTHROPIC_API_KEY else "openai",
                        "model": "claude-3-haiku-20240307" if Config.ANTHROPIC_API_KEY else "gpt-3.5-turbo",
                        "max_chunks_per_batch": 3  # Limit for cost control
                    },
                    "name": "LLM Agent"
                })
            else:
                print("Note: No LLM API keys found. LLM agent reranker will be skipped.")
        except Exception as e:
            print(f"Note: LLM components not available. LLM agent reranker will be skipped. ({e})")
        
        return rerankers
    
    def evaluate_reranker(
        self,
        reranker_config: Dict[str, Any],
        query: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a single reranker."""
        try:
            start_time = time.time()
            
            # Initialize reranker
            reranker = get_reranker(reranker_config["type"], **reranker_config["kwargs"])
            
            # Rerank results
            reranked = reranker.rerank(query, results.copy(), top_k=5)
            
            end_time = time.time()
            
            # Calculate metrics
            armenian_results = sum(1 for r in reranked[:3] if self._is_armenian_text(r["content"]))
            avg_score = sum(r.get("similarity_score", 0) for r in reranked[:3]) / min(3, len(reranked))
            
            return {
                "name": reranker_config["name"],
                "top_3_results": [r.get("display_name", f"{r['filename']}[{r.get('chunk_id', '?')}]") for r in reranked[:3]],
                "armenian_in_top_3": armenian_results,
                "avg_similarity": avg_score,
                "processing_time_ms": round((end_time - start_time) * 1000, 2),
                "total_results": len(reranked),
                "reranked_results": reranked[:3]  # Store for detailed analysis
            }
            
        except Exception as e:
            return {
                "name": reranker_config["name"],
                "error": str(e),
                "top_3_results": [],
                "armenian_in_top_3": 0,
                "avg_similarity": 0,
                "processing_time_ms": 0,
                "total_results": 0
            }
    
    def _is_armenian_text(self, text: str) -> bool:
        """Check if text contains Armenian characters."""
        armenian_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                if (0x0530 <= ord(char) <= 0x058F) or (0xFB13 <= ord(char) <= 0xFB17):
                    armenian_chars += 1
        
        return (armenian_chars / total_chars) > 0.1 if total_chars > 0 else False
    
    def run_comparison(self):
        """Run the full comparison."""
        print("🔍 Reranker Comparison Tool")
        print("=" * 50)
        
        # Load data
        sample_data = self.load_data_from_vector_store()
        if not sample_data:
            print("No data available for comparison.")
            return
        
        # Get test queries
        test_queries = self.get_test_queries()
        
        # Get available rerankers
        rerankers = self.get_available_rerankers()
        
        print(f"\nTesting {len(rerankers)} rerankers with {len(test_queries)} queries")
        print(f"Using {len(sample_data)} document chunks\n")
        
        # Run comparisons
        all_results = []
        
        for query_info in test_queries:
            query = query_info["query"]
            query_lang = query_info["language"]
            
            print(f"\n📋 Query: '{query}' ({query_info['description']})")
            print("-" * 60)
            
            query_results = []
            
            for reranker_config in rerankers:
                result = self.evaluate_reranker(reranker_config, query, sample_data)
                result["query"] = query
                result["query_language"] = query_lang
                query_results.append(result)
                all_results.append(result)
            
            # Show table for this query
            self._print_query_table(query_results)
            
            # Show detailed content for the first reranker to see what we're working with
            if query_results and "reranked_results" in query_results[0]:
                print(f"\n📄 Content Preview (using {query_results[0]['name']} reranker):")
                self._print_content_preview(query_results[0]["reranked_results"])
        
        # Show overall summary
        print(f"\n📊 OVERALL SUMMARY")
        print("=" * 70)
        self._print_summary_table(all_results)
        
        # Show consolidated comparison table
        print(f"\n📋 CONSOLIDATED COMPARISON TABLE")
        print("=" * 80)
        self._print_consolidated_table(all_results, test_queries, rerankers)
        
        # Show simplified chunk ranking table
        print(f"\n🎯 CHUNK RANKING COMPARISON")
        print("=" * 60)
        self._print_chunk_ranking_table(all_results, test_queries, rerankers)
        
        # Show detailed analysis for Armenian queries
        armenian_results = [r for r in all_results if r["query_language"] == "armenian"]
        if armenian_results:
            print(f"\n🇦🇲 ARMENIAN QUERY ANALYSIS")
            print("=" * 50)
            self._print_armenian_analysis(armenian_results)
    
    def _print_query_table(self, results: List[Dict[str, Any]]):
        """Print comparison table for a single query."""
        table_data = []
        for result in results:
            if "error" in result:
                table_data.append([
                    result["name"],
                    f"ERROR: {result['error'][:30]}...",
                    "0",
                    "0.000",
                    "0"
                ])
            else:
                # Shorten display names for better table formatting
                top_3_display = []
                for name in result["top_3_results"]:
                    if len(name) > 40:
                        # Show just filename and chunk ID
                        if '[' in name and ']' in name:
                            base_name = name.split('[')[0]
                            chunk_part = '[' + name.split('[')[1]
                            if len(base_name) > 25:
                                base_name = "..." + base_name[-22:]
                            name = base_name + chunk_part
                        else:
                            name = "..." + name[-37:]
                    top_3_display.append(name)
                
                table_data.append([
                    result["name"],
                    "\n".join(top_3_display),  # Each result on a new line for better readability
                    result["armenian_in_top_3"],
                    f"{result['avg_similarity']:.3f}",
                    f"{result['processing_time_ms']}"
                ])
        
        headers = ["Reranker", "Top 3 Chunks", "Armenian", "Avg Score", "Time (ms)"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def _print_content_preview(self, results: List[Dict[str, Any]]):
        """Print content preview of top results."""
        for i, result in enumerate(results[:3], 1):
            display_name = result.get("display_name", f"{result['filename']}[{result.get('chunk_id', '?')}]")
            content = result["content"]
            
            # Truncate content for preview
            if len(content) > 100:
                content = content[:100] + "..."
            
            print(f"  {i}. {display_name}")
            print(f"     {content}")
            print(f"     Score: {result.get('similarity_score', 0):.3f}")
            print()
    
    def _print_summary_table(self, all_results: List[Dict[str, Any]]):
        """Print overall summary table."""
        # Group by reranker
        reranker_stats = {}
        
        for result in all_results:
            name = result["name"]
            if name not in reranker_stats:
                reranker_stats[name] = {
                    "total_queries": 0,
                    "total_time": 0,
                    "total_armenian": 0,
                    "total_score": 0,
                    "errors": 0
                }
            
            stats = reranker_stats[name]
            stats["total_queries"] += 1
            
            if "error" not in result:
                stats["total_time"] += result["processing_time_ms"]
                stats["total_armenian"] += result["armenian_in_top_3"]
                stats["total_score"] += result["avg_similarity"]
            else:
                stats["errors"] += 1
        
        table_data = []
        for name, stats in reranker_stats.items():
            if stats["total_queries"] > stats["errors"]:
                avg_time = stats["total_time"] / (stats["total_queries"] - stats["errors"])
                avg_armenian = stats["total_armenian"] / (stats["total_queries"] - stats["errors"])
                avg_score = stats["total_score"] / (stats["total_queries"] - stats["errors"])
                
                table_data.append([
                    name,
                    f"{avg_time:.1f}",
                    f"{avg_armenian:.1f}",
                    f"{avg_score:.3f}",
                    stats["errors"]
                ])
            else:
                table_data.append([name, "N/A", "N/A", "N/A", stats["errors"]])
        
        headers = ["Reranker", "Avg Time (ms)", "Avg Armenian", "Avg Score", "Errors"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def _print_consolidated_table(self, all_results: List[Dict[str, Any]], test_queries: List[Dict[str, str]], rerankers: List[Dict[str, Any]]):
        """Print consolidated table with queries as rows and rerankers as columns."""
        
        # Create a mapping of query -> reranker -> result
        query_reranker_map = {}
        for result in all_results:
            query = result["query"]
            reranker = result["name"]
            
            if query not in query_reranker_map:
                query_reranker_map[query] = {}
            
            query_reranker_map[query][reranker] = result
        
        # Prepare table data
        table_data = []
        reranker_names = [r["name"] for r in rerankers]
        
        for query_info in test_queries:
            query = query_info["query"]
            query_short = query_info["description"].split(": ")[1] if ": " in query_info["description"] else query[:30]
            
            row = [f"{query_short} ({query_info['language'][:2].upper()})"]
            
            for reranker_name in reranker_names:
                if query in query_reranker_map and reranker_name in query_reranker_map[query]:
                    result = query_reranker_map[query][reranker_name]
                    
                    if "error" in result:
                        row.append("ERROR")
                    else:
                        # Show top chunk ID and score
                        if result["top_3_results"]:
                            top_chunk = result["top_3_results"][0]
                            # Extract chunk ID from display name
                            chunk_id = "?"
                            if '[' in top_chunk and ']' in top_chunk:
                                chunk_id = top_chunk.split('[')[1].split(']')[0]
                            
                            score = result["avg_similarity"]
                            time_ms = result["processing_time_ms"]
                            
                            # Format: chunk[ID] (score/time)
                            row.append(f"[{chunk_id}]\n({score:.2f}/{time_ms:.0f}ms)")
                        else:
                            row.append("No results")
                else:
                    row.append("N/A")
            
            table_data.append(row)
        
        headers = ["Query"] + reranker_names
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print("\nFormat: [chunk_id] (avg_score/time_ms)")
        print("- chunk_id: ID of top-ranked chunk")
        print("- avg_score: Average similarity score of top 3 results")
        print("- time_ms: Processing time in milliseconds")
    
    def _print_chunk_ranking_table(self, all_results: List[Dict[str, Any]], test_queries: List[Dict[str, str]], rerankers: List[Dict[str, Any]]):
        """Print simplified table showing just top chunk IDs for each query-reranker combination."""
        
        # Create a mapping of query -> reranker -> result
        query_reranker_map = {}
        for result in all_results:
            query = result["query"]
            reranker = result["name"]
            
            if query not in query_reranker_map:
                query_reranker_map[query] = {}
            
            query_reranker_map[query][reranker] = result
        
        # Prepare table data
        table_data = []
        reranker_names = [r["name"] for r in rerankers]
        
        for query_info in test_queries:
            query = query_info["query"]
            # Create shorter query description
            lang_code = query_info["language"][:2].upper()
            if ": " in query_info["description"]:
                query_short = query_info["description"].split(": ")[1][:25]
            else:
                query_short = query[:25]
            
            if len(query_short) > 25:
                query_short = query_short[:22] + "..."
            
            row = [f"{query_short} ({lang_code})"]
            
            for reranker_name in reranker_names:
                if query in query_reranker_map and reranker_name in query_reranker_map[query]:
                    result = query_reranker_map[query][reranker_name]
                    
                    if "error" in result:
                        row.append("ERR")
                    else:
                        # Show just the top chunk ID
                        if result["top_3_results"]:
                            top_chunk = result["top_3_results"][0]
                            # Extract chunk ID from display name
                            chunk_id = "?"
                            if '[' in top_chunk and ']' in top_chunk:
                                chunk_id = top_chunk.split('[')[1].split(']')[0]
                            row.append(f"[{chunk_id}]")
                        else:
                            row.append("[-]")
                else:
                    row.append("N/A")
            
            table_data.append(row)
        
        headers = ["Query"] + [name.replace(" ", "\n") for name in reranker_names]  # Multi-line headers
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print("\nLegend:")
        print("- [XX]: Chunk ID of top-ranked result")
        print("- ERR: Error occurred during reranking")
        print("- [-]: No results returned")
        print("- Different chunk IDs indicate different reranking behavior")
    
    def _print_armenian_analysis(self, armenian_results: List[Dict[str, Any]]):
        """Print detailed analysis for Armenian queries."""
        print("Performance with Armenian queries:")
        
        # Group by reranker
        reranker_armenian = {}
        for result in armenian_results:
            name = result["name"]
            if name not in reranker_armenian:
                reranker_armenian[name] = []
            reranker_armenian[name].append(result["armenian_in_top_3"])
        
        table_data = []
        for name, armenian_counts in reranker_armenian.items():
            avg_armenian = sum(armenian_counts) / len(armenian_counts)
            max_armenian = max(armenian_counts)
            table_data.append([
                name,
                f"{avg_armenian:.1f}",
                f"{max_armenian}",
                len([x for x in armenian_counts if x > 0])
            ])
        
        headers = ["Reranker", "Avg Armenian/Top3", "Max Armenian", "Queries w/Armenian"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        best_for_armenian = max(reranker_armenian.items(), 
                               key=lambda x: sum(x[1]) / len(x[1]))
        print(f"• Best for Armenian content: {best_for_armenian[0]}")
        print("• Cross-encoder models provide more accurate semantic matching")
        print("• Multilingual cross-encoder specifically boosts Armenian content")


def main():
    """Main function."""
    try:
        comparator = RerankerComparator()
        comparator.run_comparison()
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user.")
    except Exception as e:
        print(f"\nError during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
